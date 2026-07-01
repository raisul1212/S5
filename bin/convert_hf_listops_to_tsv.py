"""Convert HuggingFace fengyang0317/listops-1000 to LRA-format TSVs
that S5's dataloader consumes unchanged.

Google's original storage.googleapis.com/long-range-arena/lra_release.gz
is 403 (bucket taken private).  fengyang0317/listops-1000 is a
faithful mirror of the same underlying Nangia-Bowman ListOps
generation used by LRA's listops-1000 subset -- same {Source, Target}
schema, same 96K/2K/2K split.

Output files at self.data_dir/basic_{train,val,test}.tsv, format:
  Source<TAB>Target
  <bracket-string><TAB><int>
  ...

S5's ListOps.process_dataset reads these via
  load_dataset('csv', data_files={'train':..., 'val':..., 'test':...},
               delimiter='\\t')
and then applies listops_tokenizer (token-level split) + vocab build.
"""
import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset


def write_tsv(examples, path):
    """Write examples to a TSV with 'Source\\tTarget' header, one example per row."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("Source\tTarget\n")
        for ex in examples:
            source = ex["Source"]
            target = ex["Target"]
            # Sanity: source shouldn't contain tabs or newlines (LRA convention)
            if "\t" in source or "\n" in source:
                raise ValueError(f"Source contains delimiter: {source[:80]!r}...")
            f.write(f"{source}\t{target}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for basic_{train,val,test}.tsv.  "
                         "S5 expects: raw_datasets/lra_release/lra_release/listops-1000/")
    ap.add_argument("--hf-name", default="fengyang0317/listops-1000",
                    help="HuggingFace dataset name to source from")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading HF dataset: {args.hf_name}")
    ds = load_dataset(args.hf_name)
    print(f"[*] Splits: {list(ds.keys())}")

    # HF: {'train': 96000, 'validation': 2000, 'test': 2000}
    # S5 expects file names: basic_train / basic_val / basic_test
    split_map = {
        "train": "basic_train.tsv",
        "validation": "basic_val.tsv",
        "test": "basic_test.tsv",
    }

    for hf_split, tsv_name in split_map.items():
        if hf_split not in ds:
            raise KeyError(f"HF dataset missing split {hf_split!r}; got {list(ds.keys())}")
        n = len(ds[hf_split])
        out_path = out_dir / tsv_name
        print(f"[*] Writing {tsv_name}: {n:,} examples -> {out_path}")
        write_tsv(ds[hf_split], out_path)

    # Print sizes for sanity
    print("\n[done] Output files:")
    for tsv_name in split_map.values():
        p = out_dir / tsv_name
        n_lines = sum(1 for _ in open(p, "r", encoding="utf-8"))
        size_mb = p.stat().st_size / (1024 * 1024)
        # -1 for the header row
        print(f"  {p}: {n_lines-1:,} data rows, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
