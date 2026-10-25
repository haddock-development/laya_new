"""Tokenize flywheel decision-cases into laya items (reuses pilot pipeline).

Input : mini-eureka docs/evidence/decide-flywheel/cases-pairs.jsonl
Output: work/pilot/items_flywheel_{split}.pt
Splits are assigned per GROUP (source journal + steps) so pairs sharing the
same recipes never straddle splits.

Usage: python work/pilot/build_flywheel_items.py --cases <cases.jsonl>
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from transformers import AutoTokenizer  # noqa: E402
from laya.common import QTYPES, build_sequence  # noqa: E402

BASE = os.path.join(HERE, "..", "laya-base")
MAX_LEN = int(os.environ.get("LAYA_MAXLEN", "1024"))
HEAD_LEN = int(os.environ.get("LAYA_HEAD", "512"))


def split_assign(cases: list[dict]) -> None:
    """Case-Level-Split nach case_id-Hash (70/10/10/10).

    Bewusst KEIN Rezept-Group-Split: Rezept->Qualitaet ist genau das
    Signal, das gelernt werden soll; dasselbe Rezept darf in mehreren
    Splits auftauchen. Der echte Haertest sind neue Studien.
    """
    for c in cases:
        h = int(hashlib.sha256(c["case_id"].encode()).hexdigest(), 16) % 100
        c["split"] = ("train" if h < 70 else "val" if h < 80
                      else "calibration" if h < 90 else "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--splits", default="train,val,calibration,test")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(os.path.join(BASE, "tokenizer"))
    cases = [json.loads(l) for l in open(a.cases) if l.strip()]
    split_assign(cases)
    labels = ["A", "B"]
    for split in a.splits.split(","):
        items = []
        for c in cases:
            if c["split"] != split:
                continue
            q = {"t": "choice", "ins": c["instructions"],
                 "crit": dict(c["criteria"])}
            state = c["state"]
            seq, markers = build_sequence(tok, state, q, MAX_LEN, HEAD_LEN)
            gold_idx = labels.index(c["gold"])
            target = [0.0] * len(markers)
            target[gold_idx] = 1.0
            items.append({"ids": seq, "markers": markers, "target": target,
                          "qid": "flywheel", "gold": c["gold"],
                          "qtype": QTYPES["choice"], "label": gold_idx,
                          "case_id": c["case_id"]})
        out = os.path.join(HERE, "items_flywheel_%s.pt" % split)
        import torch
        torch.save(items, out)
        lens = sorted(len(i["ids"]) for i in items)
        print("%-28s %4d items  p50=%d p95=%d max=%d" % (
            os.path.basename(out), len(items), lens[len(lens) // 2],
            lens[int(len(lens) * 0.95)], lens[-1]), flush=True)


if __name__ == "__main__":
    main()

