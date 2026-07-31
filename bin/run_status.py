"""Live run tracker -- emits RUNS.md + RUNS.csv for every Mambino job.

Joins SLURM (squeue for live, sacct for finished) against the checkpoint dirs,
which are always suffixed with the job id, so no manifest is needed.

Job-name convention (set at submit time, or patched with `scontrol update
jobid=N JobName=...`):   <MODEL>.<PROTOCOL>.<SEED>      e.g. GF.e200d0.42
    MODEL     S5 | 0 | G | F | Fo | GF     (see paper_v2_ppac/memos/NAMING_AND_STATUS.md)
    PROTOCOL  e<epochs>d<dropout>          e40d0, e200d0, e200d0.235

Usage (on Gilbreth, from the worktree root):
    /scratch/gilbreth/raisul/envs/s5m/bin/python bin/run_status.py
Writes RUNS.md and RUNS.csv next to it and prints the table.
"""
import csv
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")

# canonical model names -- keep in sync with NAMING_AND_STATUS.md
MODEL = {"S5": "Pure S5 (Corner 1)", "0": "Mambino-0", "G": "Mambino-G",
         "F": "Mambino-F", "Fo": "Mambino-F(fixed)", "GF": "Mambino-GF"}
PARAMS = {"S5": 188490, "0": 105738, "G": 105754, "F": 123682,
          "Fo": 123682, "GF": 123698}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def jobs():
    """job_id -> (name, state, elapsed) from squeue (live) + sacct (finished)."""
    out = {}
    for ln in sh('sacct -u $USER --starttime now-7days -X -n -P '
                 '-o JobID,JobName,State,Elapsed').splitlines():
        f = ln.split("|")
        if len(f) >= 4 and f[0].isdigit():
            out[f[0]] = (f[1], f[2].split()[0], f[3])
    for ln in sh('squeue -u $USER -h -o "%i|%j|%T|%M"').splitlines():
        f = ln.split("|")
        if len(f) >= 4 and f[0].isdigit():
            out[f[0]] = (f[1], f[2], f[3])          # squeue wins (fresher)
    return out


def progress(job_id):
    """(epochs_done, test@peakval_so_far, ckpt_dir) by globbing *_<jobid>."""
    d = glob.glob(os.path.join(CKPT, f"*_{job_id}"))
    if not d or not os.path.exists(os.path.join(d[0], "run.log")):
        return 0, None, ""
    v, t = [], []
    for ln in open(os.path.join(d[0], "run.log"), errors="ignore"):
        if "Train Loss:" in ln:
            a = re.search(r"Val Accuracy:\s*([0-9.]+)", ln)
            b = re.search(r"Test Accuracy:\s*([0-9.]+)", ln)
            if a and b:
                v.append(float(a.group(1)))
                t.append(float(b.group(1)))
    if not v:
        return 0, None, os.path.basename(d[0])
    return len(v), t[v.index(max(v))], os.path.basename(d[0])


def parse(name):
    """<MODEL>.<PROTOCOL>.<SEED> -> (model, epochs, dropout, seed)."""
    m = re.match(r"^([A-Za-z0-9]+)\.e(\d+)d([0-9.]+)\.(\S+)$", name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3), m.group(4)


def main():
    rows = []
    for jid, (name, state, elapsed) in jobs().items():
        p = parse(name)
        if not p:                                    # foreign job (e.g. enwik8)
            rows.append(dict(job=jid, model="-", label=name, seed="-", epochs="-",
                             dropout="-", state=state, elapsed=elapsed, done="-",
                             total="-", acc="", ckpt=""))
            continue
        mdl, ep, do, seed = p
        n, acc, ck = progress(jid)
        rows.append(dict(job=jid, model=MODEL.get(mdl, mdl), label=mdl, seed=seed,
                         epochs=ep, dropout=do, state=state, elapsed=elapsed,
                         done=n, total=ep, acc=f"{acc:.4f}" if acc else "",
                         ckpt=ck))
    order = {"RUNNING": 0, "PENDING": 1, "COMPLETED": 2}
    rows.sort(key=lambda r: (order.get(r["state"], 3), str(r["model"]),
                             str(r["epochs"]), str(r["seed"])))

    with open(os.path.join(ROOT, "RUNS.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = (f"| job | model | params | seed | ep | drop | state | elapsed |"
           f" progress | test@peakval |")
    sep = "|---|---|---:|---|---:|---:|---|---:|---|---:|"
    stamp = sh("date '+%Y-%m-%d %H:%M %Z'")
    host = sh("hostname")
    out = ["# Mambino run status", "",
           f"Generated: {stamp} on {host}",
           "", "Model names per `paper_v2_ppac/memos/NAMING_AND_STATUS.md`.", "",
           hdr, sep]
    for r in rows:
        pr = f"{r['done']}/{r['total']}" if r["total"] != "-" else "-"
        pm = PARAMS.get(r["label"], "")
        out.append(f"| {r['job']} | {r['model']} | {pm or ''} | {r['seed']} |"
                   f" {r['epochs']} | {r['dropout']} | {r['state']} |"
                   f" {r['elapsed']} | {pr} | {r['acc']} |")
    md = "\n".join(out) + "\n"
    open(os.path.join(ROOT, "RUNS.md"), "w").write(md)
    print(md)
    print(f"wrote {os.path.join(ROOT,'RUNS.md')} and RUNS.csv")


if __name__ == "__main__":
    main()
