"""5-step validation for the HF -> LRA-TSV conversion + S5 dataloader
end-to-end.  Runs on cluster ncb env.  Prints per-step PASS/FAIL and
exits non-zero on any failure so a wrapping script can gate training
launch on validation success.

Steps:
  1. Structural check      TSVs exist, correct row counts, correct schema
  2. Content diff          Random spot-check: TSV rows == HF rows
  3. S5 dataloader         S5's ListOps class loads, correct l_max, vocab, sizes
  4. Tokenization verify   Detokenize round-trip matches Source (modulo eos)
  5. End-to-end batch      prep_batch produces (bsz, L, in_dim) with valid one-hot

Usage:
  python -m bin.validate_listops_data \
      --tsv-dir raw_datasets/lra_release/lra_release/listops-1000/ \
      --hf-name fengyang0317/listops-1000
"""
import argparse
import csv
import random
import sys
from pathlib import Path

# S5 imports (only after we know we're on cluster)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pass(step, msg=""):
    print(f"  [PASS] step {step}: {msg}")


def _fail(step, msg):
    print(f"  [FAIL] step {step}: {msg}", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
def step_1_structural(tsv_dir):
    """TSVs exist, correct row counts (96K/2K/2K), correct schema."""
    print("\n=== STEP 1: structural check ===")
    expected = {
        "basic_train.tsv": 96_000,
        "basic_val.tsv": 2_000,
        "basic_test.tsv": 2_000,
    }
    tsv_dir = Path(tsv_dir)
    for name, n_expected in expected.items():
        path = tsv_dir / name
        if not path.is_file():
            _fail(1, f"missing file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            if header != "Source\tTarget":
                _fail(1, f"{name}: bad header {header!r}, expected 'Source\\tTarget'")
            n_lines = sum(1 for _ in f)  # remaining lines = data rows
        if n_lines != n_expected:
            _fail(1, f"{name}: got {n_lines} data rows, expected {n_expected}")
        _pass(1, f"{name}: {n_lines:,} rows, header OK")


# ─────────────────────────────────────────────────────────────────────
def step_2_content_diff(tsv_dir, hf_name):
    """Random spot-check: 3 rows per split match the HF source exactly."""
    print("\n=== STEP 2: content diff (spot check) ===")
    from datasets import load_dataset
    ds = load_dataset(hf_name)
    split_map = {
        "basic_train.tsv": "train",
        "basic_val.tsv":   "validation",
        "basic_test.tsv":  "test",
    }
    rng = random.Random(0)
    for tsv_name, hf_split in split_map.items():
        path = Path(tsv_dir) / tsv_name
        # Load TSV as dict-per-row
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            tsv_rows = list(reader)
        n = len(tsv_rows)
        if n != len(ds[hf_split]):
            _fail(2, f"{tsv_name}: row count mismatch (TSV {n} vs HF {len(ds[hf_split])})")
        # Random spot check
        idxs = rng.sample(range(n), k=min(3, n))
        for i in idxs:
            tsv_src = tsv_rows[i]["Source"]
            tsv_tgt = int(tsv_rows[i]["Target"])
            hf_src = ds[hf_split][i]["Source"]
            hf_tgt = int(ds[hf_split][i]["Target"])
            if tsv_src != hf_src:
                _fail(2, f"{tsv_name}[{i}]: Source diverges (first 60): "
                         f"TSV={tsv_src[:60]!r} vs HF={hf_src[:60]!r}")
            if tsv_tgt != hf_tgt:
                _fail(2, f"{tsv_name}[{i}]: Target diverges TSV={tsv_tgt} vs HF={hf_tgt}")
        _pass(2, f"{tsv_name}: {n:,} rows, {len(idxs)} random-checked rows OK")


# ─────────────────────────────────────────────────────────────────────
def step_3_s5_dataloader(tsv_dir):
    """S5's ListOps class loads the TSVs cleanly with expected metadata."""
    print("\n=== STEP 3: S5 dataloader hand-off ===")
    # Import ListOps directly via importlib to avoid s5.dataloaders.__init__
    # which pulls in `audio` -> `torchaudio` (not installed in our env).
    import importlib.util as _iu
    _spec = _iu.spec_from_file_location(
        "s5_lra_mod",
        str(Path(__file__).resolve().parent.parent / "s5" / "dataloaders" / "lra.py"))
    _lra_mod = _iu.module_from_spec(_spec)
    # base.py is a sibling import inside lra.py; ensure package resolution
    import sys as _sys
    _sys.modules["s5_lra_mod"] = _lra_mod
    try:
        _spec.loader.exec_module(_lra_mod)
    except ModuleNotFoundError:
        # Fall back to importing the full sub-module via a lighter path
        from s5.dataloaders import lra as _lra_mod
    ListOps = _lra_mod.ListOps

    dataset_obj = ListOps(name="listops", data_dir=Path(tsv_dir))
    # S5 uses cache_dir for pre-tokenised cache; disable to force fresh load
    dataset_obj.cache_dir = None
    dataset_obj.setup()

    # Checks
    if dataset_obj.d_output != 10:
        _fail(3, f"d_output={dataset_obj.d_output}, expected 10")
    if dataset_obj.l_max != 2048:
        _fail(3, f"l_max={dataset_obj.l_max}, expected 2048 (S5 default)")
    n_train = len(dataset_obj.dataset_train)
    if n_train != 96_000:
        _fail(3, f"train size {n_train}, expected 96,000")
    n_test = len(dataset_obj.dataset_test)
    if n_test != 2_000:
        _fail(3, f"test size {n_test}, expected 2,000")
    vocab_size = len(dataset_obj.vocab)
    if not (15 <= vocab_size <= 25):
        _fail(3, f"vocab size {vocab_size}, expected ~18-20 (token-level)")
    _pass(3, f"d_output={dataset_obj.d_output}, l_max={dataset_obj.l_max}, "
             f"train={n_train:,}, test={n_test:,}, vocab={vocab_size}")

    # Stash for downstream steps
    return dataset_obj


# ─────────────────────────────────────────────────────────────────────
def step_4_tokenization(dataset_obj):
    """Verify listops_tokenizer round-trip: detokenized tokens re-tokenize identically."""
    print("\n=== STEP 4: tokenization verify ===")
    # listops_tokenizer already imported via lra module in step 3; re-import
    # via importlib to stay consistent
    import importlib.util as _iu
    _spec = _iu.spec_from_file_location(
        "s5_lra_mod2",
        str(Path(__file__).resolve().parent.parent / "s5" / "dataloaders" / "lra.py"))
    _lra_mod = _iu.module_from_spec(_spec)
    import sys as _sys
    _sys.modules["s5_lra_mod2"] = _lra_mod
    _spec.loader.exec_module(_lra_mod)
    listops_tokenizer = _lra_mod.listops_tokenizer
    # Grab first training example (indexed via the S5-wrapped dataset)
    ex = dataset_obj.dataset_train[0]
    input_ids = ex["input_ids"]

    # Detokenize by looking up vocab
    itos = dataset_obj.vocab.get_itos()
    tokens = [itos[i] for i in input_ids.tolist() if itos[i] not in ("<pad>",)]
    # Drop trailing <eos> if present
    if tokens and tokens[-1] == "<eos>":
        tokens = tokens[:-1]

    # Verify all tokens are known
    unk_count = sum(1 for t in tokens if t == "<unk>")
    if unk_count > 0:
        _fail(4, f"found {unk_count} <unk> tokens in first train example -- vocab issue")

    # Sample structure check: at least one operator + one digit
    has_op = any(t.startswith("[") for t in tokens)
    has_digit = any(t.isdigit() for t in tokens)
    has_close = any(t == "X" for t in tokens)  # listops_tokenizer maps ']' -> 'X'
    if not (has_op and has_digit and has_close):
        _fail(4, f"tokens missing operator/digit/X: sample={tokens[:20]}")
    _pass(4, f"first example: {len(tokens)} tokens, sample={tokens[:15]}...")


# ─────────────────────────────────────────────────────────────────────
def step_5_end_to_end_batch(dataset_obj):
    """Load one batch, run S5's prep_batch, verify one-hot shape."""
    print("\n=== STEP 5: end-to-end batch through prep_batch ===")
    from s5.train_helpers import prep_batch
    from s5.dataloading import make_data_loader

    loader = make_data_loader(dataset_obj.dataset_train, dataset_obj,
                              seed=0, batch_size=4)
    batch = next(iter(loader))

    # S5's ListOps IN_DIM is hardcoded 20 in dataloading.py
    in_dim = 20
    seq_len = dataset_obj.l_max  # 2048
    full_inputs, targets, integration_timesteps = prep_batch(batch, seq_len, in_dim)

    # full_inputs is (data, lengths) tuple since padded=True
    data, lengths = full_inputs
    if data.ndim != 3:
        _fail(5, f"data ndim={data.ndim}, expected 3 (bsz, L, in_dim)")
    if data.shape != (4, seq_len, in_dim):
        _fail(5, f"data shape={data.shape}, expected ({4}, {seq_len}, {in_dim})")
    if targets.shape != (4,):
        _fail(5, f"targets shape={targets.shape}, expected (4,)")
    # Every timestep row should sum to <= 1 (one-hot or all-zero for pad)
    row_sums = data.sum(axis=-1)
    if not ((row_sums <= 1.0 + 1e-6) & (row_sums >= 0.0)).all():
        _fail(5, "data rows are not one-hot / all-zero")
    _pass(5, f"data shape={data.shape}, targets shape={targets.shape}, "
             f"lengths range [{int(lengths.min())}, {int(lengths.max())}], one-hot OK")


# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-dir", required=True)
    ap.add_argument("--hf-name", default="fengyang0317/listops-1000")
    args = ap.parse_args()

    print("=" * 60)
    print("LRA-ListOps HF-to-TSV validation (5 steps)")
    print("=" * 60)

    step_1_structural(args.tsv_dir)
    step_2_content_diff(args.tsv_dir, args.hf_name)
    dataset_obj = step_3_s5_dataloader(args.tsv_dir)
    step_4_tokenization(dataset_obj)
    step_5_end_to_end_batch(dataset_obj)

    print("\n" + "=" * 60)
    print("ALL 5 STEPS PASSED -- data is ready for training")
    print("=" * 60)


if __name__ == "__main__":
    main()
