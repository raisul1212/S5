"""Stage C, module 1 — the cross-lingual byte stream (pure, no jax; testable in isolation).

Builds a single contiguous byte stream English -> L2 -> English with KNOWN shift boundaries,
which the streaming eval scores position-by-position (the recovery curve).
"""
import numpy as np


def load_bytes(path, n=None, offset=0):
    """Read a file as a uint8->int64 byte array (byte-level LM, vocab 256)."""
    with open(path, "rb") as f:
        b = f.read()
    b = b[offset:] if n is None else b[offset:offset + n]
    if n is not None:
        assert len(b) == n, f"{path}: wanted {n} bytes at offset {offset}, got {len(b)} (file too short)"
    return np.frombuffer(b, dtype=np.uint8).astype(np.int64)


def build_stream(segments):
    """segments: list of (label:str, byte_array:int64[]).
    Returns dict with:
      ids       : int64[L]           concatenated bytes
      labels    : object[L]          per-position segment label
      seg_id    : int32[L]           per-position segment index (0,1,2,...)
      bounds    : list[int]          positions where a new segment STARTS (shift points)
      spans     : list[(label,a,b)]  (label, start, end) per segment
    """
    ids, labels, seg_id, bounds, spans = [], [], [], [], []
    pos = 0
    for i, (label, arr) in enumerate(segments):
        arr = np.asarray(arr, dtype=np.int64)
        if i > 0:
            bounds.append(pos)
        ids.append(arr)
        labels += [label] * len(arr)
        seg_id += [i] * len(arr)
        spans.append((label, pos, pos + len(arr)))
        pos += len(arr)
    return dict(ids=np.concatenate(ids), labels=np.array(labels, dtype=object),
                seg_id=np.array(seg_id, dtype=np.int32), bounds=bounds, spans=spans)


def eng_l2_eng(eng_path, l2_path, seg_bytes, eng_offset=0, l2_offset=0, l2_label="L2"):
    """Convenience: [English seg][L2 seg][English seg] each `seg_bytes` long.
    English segments are DISJOINT slices (so the 2nd English isn't a repeat of the 1st)."""
    eng = load_bytes(eng_path, n=2 * seg_bytes, offset=eng_offset)
    l2 = load_bytes(l2_path, n=seg_bytes, offset=l2_offset)
    return build_stream([("EN", eng[:seg_bytes]), (l2_label, l2), ("EN2", eng[seg_bytes:2 * seg_bytes])])
