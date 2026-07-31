"""Live run tracker -- emits RUNS.md + RUNS.csv for every Mambino job.

Joins SLURM (squeue live + sacct history) against checkpoint dirs, which are
always suffixed with the job id.  Config is resolved two ways, in order:

  1. job name, convention <MODEL>.<PROTOCOL>.<SEED>   e.g. GF.e200d0.42
     (set at submit time, or patched: scontrol update jobid=N JobName=...)
  2. checkpoint DIRECTORY name -- so jobs submitted before the naming
     convention existed still resolve.

Searches every Mambino worktree, since Cluster-A runs live in S5-sgate/S5-sgate2.

Model labels per paper_v2_ppac/memos/NAMING_AND_STATUS.md:
    S5 Pure S5 (Corner 1) | 0 Mambino-0 | G Mambino-G
    F Mambino-F (adaptive FWM write) | Fo Mambino-F(fixed) | GF Mambino-GF

Usage on Gilbreth, from the worktree root:
    /scratch/gilbreth/raisul/envs/s5m/bin/python bin/run_status.py
"""
import csv
import glob
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
CKPT_ROOTS = [os.path.join(ROOT, "checkpoints")] + [
    f"{HOME}/dev/ssm-baselines/{w}/checkpoints"
    for w in ("S5-clusterb", "S5-sgate", "S5-sgate2", "S5-charlm")]

MODEL = {"S5": "Pure S5 (Corner 1)", "0": "Mambino-0", "G": "Mambino-G",
         "F": "Mambino-F", "Fo": "Mambino-F(fix)", "GF": "Mambino-GF"}
PARAMS = {"S5": 188490, "0": 105738, "G": 105754, "F": 123682,
          "Fo": 123682, "GF": 123698}
# s7proto CONFIG token -> model label
S7MAP = {"corner1": "S5", "clusterA": "G", "clusterB": "GF"}
# fast-weight ARM token -> model label
FWMAP = {"surprise": "F", "const": "Fo", "stacked": "GF"}
MAXDONE = 30                                    # cap COMPLETED rows shown


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def slurm_jobs():
    out = {}
    for ln in sh("sacct -u $USER --starttime now-14days -X -n -P "
                 "-o JobID,JobName,State,Elapsed").splitlines():
        f = ln.split("|")
        if len(f) >= 4 and f[0].isdigit():
            out[f[0]] = [f[1], f[2].split()[0], f[3]]
    for ln in sh('squeue -u $USER -h -o "%i|%j|%T|%M"').splitlines():
        f = ln.split("|")
        if len(f) >= 4 and f[0].isdigit():
            out[f[0]] = [f[1], f[2], f[3]]        # squeue is fresher
    return out


def ckpt_dir(job_id):
    for r in CKPT_ROOTS:
        d = glob.glob(os.path.join(r, f"*_{job_id}"))
        if d:
            return d[0]
    return ""


def from_dirname(b):
    """checkpoint dir basename -> (model, epochs, dropout, seed) or None."""
    m = re.match(r"s7proto_(\w+)_e(\d+)_do([0-9.]+)_s(\S+?)_\d+$", b)
    if m:
        return S7MAP.get(m.group(1), m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    m = re.match(r"mambino_gelu_fw_(\w+?)_d(\d+)_s(\S+?)_\d+$", b)
    if m:
        return FWMAP.get(m.group(1), m.group(1)), 40, "0", m.group(3)
    m = re.match(r"mambino_gelu_sgate_(signed|unsigned)_\S*?_s?(\S+?)_\d+$", b)
    if m:
        return "G", 40, "0", m.group(2)
    return None


def from_jobname(name):
    m = re.match(r"^([A-Za-z0-9]+)\.e(\d+)d([0-9.]+)\.(\S+)$", name)
    return (m.group(1), int(m.group(2)), m.group(3), m.group(4)) if m else None


def progress(d):
    if not d or not os.path.exists(os.path.join(d, "run.log")):
        return 0, None
    v, t = [], []
    for ln in open(os.path.join(d, "run.log"), errors="ignore"):
        if "Train Loss:" in ln:
            a = re.search(r"Val Accuracy:\s*([0-9.]+)", ln)
            b = re.search(r"Test Accuracy:\s*([0-9.]+)", ln)
            if a and b:
                v.append(float(a.group(1)))
                t.append(float(b.group(1)))
    return (len(v), t[v.index(max(v))]) if v else (0, None)


def main():
    rows, foreign = [], []
    for jid, (name, state, elapsed) in slurm_jobs().items():
        d = ckpt_dir(jid)
        cfg = from_jobname(name) or (from_dirname(os.path.basename(d)) if d else None)
        if not cfg:
            foreign.append(f"{jid} {name} ({state})")
            continue
        mdl, ep, do, seed = cfg
        n, acc = progress(d)
        rows.append(dict(job=jid, label=mdl, model=MODEL.get(mdl, mdl),
                         params=PARAMS.get(mdl, ""), seed=seed, epochs=ep,
                         dropout=do, state=state, elapsed=elapsed, done=n,
                         acc=f"{acc:.4f}" if acc is not None else "",
                         ckpt=os.path.basename(d)))
    rank = {"RUNNING": 0, "PENDING": 1}
    rows.sort(key=lambda r: (rank.get(r["state"], 2), -int(r["job"])))
    shown = [r for r in rows if r["state"] in ("RUNNING", "PENDING")] + \
            [r for r in rows if r["state"] not in ("RUNNING", "PENDING")][:MAXDONE]

    if rows:
        with open(os.path.join(ROOT, "RUNS.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    o = ["# Mambino run status", "",
         f"Generated: {sh(chr(100)+'ate ' + chr(39) + '+%Y-%m-%d %H:%M %Z' + chr(39))}",
         "", "Model names per `paper_v2_ppac/memos/NAMING_AND_STATUS.md`.",
         f"Showing all live jobs + the {MAXDONE} most recent finished.", "",
         "| job | model | params | seed | ep | drop | state | elapsed | progress | test@peakval |",
         "|---|---|---:|---|---:|---:|---|---:|---|---:|"]
    for r in shown:
        o.append(f"| {r['job']} | {r['model']} | {r['params']} | {r['seed']} |"
                 f" {r['epochs']} | {r['dropout']} | {r['state']} | {r['elapsed']} |"
                 f" {r['done']}/{r['epochs']} | {r['acc']} |")
    if foreign:
        o += ["", "Other jobs (not Mambino runs): " + ", ".join(foreign[:8])]
    md = "\n".join(o) + "\n"
    open(os.path.join(ROOT, "RUNS.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
