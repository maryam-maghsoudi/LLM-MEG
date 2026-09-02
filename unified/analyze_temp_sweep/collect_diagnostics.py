#!/usr/bin/env python3
"""
collect_diagnostics.py — Collect embedding diagnostics across the temperature sweep.

For every (temperature × heldout subject) checkpoint, computes scalar diagnostics for:
  - unseen      : the LOSO heldout subject
  - seen_single : one fixed training subject (first non-heldout alphabetically)
  - seen_avg    : mean of all 12 non-heldout training subjects

MEG data is loaded ONCE per subject (from icaed_Sai npy files) and reused
across all 8 temperatures, so total MEG loads = 13 (one per subject).
BERT is also loaded once.

Output: analyze_temp_sweep/sweep_diagnostics.json

Run from llm_decoder/:
    python -m unified.analyze_temp_sweep.collect_diagnostics
    python -m unified.analyze_temp_sweep.collect_diagnostics --device cuda
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_LLMDEC = Path(__file__).resolve().parent.parent.parent   # llm_decoder/
sys.path.insert(0, str(_LLMDEC))

from unified.data.base_dataset import MEGWordDataset, ONSET_DIR
from unified.data.splits import POEM_KEYS
from unified.methods.models import MEGEncoder, BERTTextProjection, load_bert_hiddens
from unified.method1_analysis.visualize_embeddings import (
    effective_rank, pairwise_cos_stats, per_trial_cos_stats,
    compute_nn_purity, compute_diagnostics,
)

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09",
    "sub-10", "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]

SCALAR_METRICS = [
    "meg_effective_rank", "text_effective_rank",
    "meg_pairwise_cos_mean", "meg_pairwise_cos_std",
    "text_pairwise_cos_mean", "text_pairwise_cos_std",
    "intra_trial_cos_mean", "intra_trial_cos_std",
    "nn_exact_match", "nn_purity_top5",
]

PREPROC = Path("/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed_Sai")


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect embedding diagnostics across the temperature sweep"
    )
    p.add_argument("--sweep_dir",
                   default=str(_LLMDEC / "unified/sweep_temp_contrastive"),
                   help="Root of temperature sweep directories")
    p.add_argument("--out_dir",
                   default=str(_LLMDEC / "unified/analyze_temp_sweep"),
                   help="Directory to write sweep_diagnostics.json")
    p.add_argument("--model_tag", default="bert_base_uncased")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--bert_name", default="bert-base-uncased")
    p.add_argument("--bert_layer", type=int, default=-1)
    p.add_argument("--nn_k", type=int, default=5, help="k for NN purity")
    p.add_argument("--sessions", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--poems", nargs="+", default=list(POEM_KEYS))
    return p.parse_args()


# ─── Raw MEG data loading ─────────────────────────────────────────────────────

def load_raw_subject(
    subject: str,
    sessions: List[int],
    poems: List[str],
    bert_hiddens: Dict[str, torch.Tensor],
    meg_base: Optional[Path],
) -> Optional[Dict]:
    """
    Load raw MEG windows + BERT targets for one subject into CPU tensors.
    Returns None if no valid windows found.
    """
    trials = [(subject, poem, sess) for poem in poems for sess in sessions]
    try:
        ds = MEGWordDataset(trials, augment=False, meg_base=meg_base)
    except Exception as e:
        print(f"    [warn] {subject}: dataset error — {e}")
        return None

    if len(ds) == 0:
        print(f"    [warn] {subject}: 0 valid windows")
        return None

    meg_list, bert_list = [], []
    word_texts, word_poses, all_poems, all_sessions, all_trials = [], [], [], [], []

    for i in range(len(ds)):
        item = ds[i]
        raw  = ds._items[i]
        meg_list.append(item["meg_window"])
        bert_list.append(bert_hiddens[raw["poem"]][raw["word_pos"]])
        word_texts.append(raw["word_text"])
        word_poses.append(raw["word_pos"])
        all_poems.append(raw["poem"])
        all_sessions.append(raw["session"])
        all_trials.append((raw["poem"], raw["session"]))

    trial_groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, t in enumerate(all_trials):
        trial_groups[t].append(i)

    return {
        "meg":          torch.stack(meg_list),   # (N, 155, 40)
        "bert":         torch.stack(bert_list),  # (N, 768)
        "word_text":    word_texts,
        "word_pos":     word_poses,
        "poem":         all_poems,
        "session":      all_sessions,
        "trial_groups": dict(trial_groups),
    }


# ─── Precomputed vocab BERT hiddens ──────────────────────────────────────────

def build_h_vocab(
    bert_hiddens: Dict[str, torch.Tensor],
    poems: List[str],
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute mean-pooled BERT hidden per unique word type across all poems.
    Returns (h_vocab: (V, 768), vocab_words: List[str]).  Fixed across checkpoints.
    """
    word_to_hs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for poem in poems:
        onset_path = ONSET_DIR / f"{poem}_word_onsets.json"
        entries = json.loads(onset_path.read_text())
        h_poem  = bert_hiddens[poem]
        for i, e in enumerate(entries):
            word = e["word"].strip().lower()
            word_to_hs[word].append(h_poem[i])

    vocab_words = sorted(word_to_hs.keys())
    h_vocab = torch.stack([
        torch.stack(word_to_hs[w]).mean(0) for w in vocab_words
    ])  # (V, 768)
    return h_vocab, vocab_words


# ─── Forward pass + diagnostics ──────────────────────────────────────────────

def run_diagnostics(
    raw:         Dict,
    meg_enc:     MEGEncoder,
    bert_proj:   BERTTextProjection,
    h_vocab:     torch.Tensor,    # (V, 768) fixed
    vocab_words: List[str],
    device:      torch.device,
    k:           int = 5,
) -> Dict:
    """Run encoder forward pass on pre-loaded tensors and return diagnostics dict."""
    meg_enc.eval()
    bert_proj.eval()

    with torch.no_grad():
        z_meg  = meg_enc(raw["meg"].to(device)).cpu()              # (N, 128)
        z_text = bert_proj(raw["bert"].float().to(device)).cpu()   # (N, 128)
        z_text_vocab = bert_proj(h_vocab.float().to(device)).cpu() # (V, 128)

    data = {
        "z_meg":        z_meg,
        "z_text":       z_text,
        "z_text_vocab": z_text_vocab,
        "vocab_words":  vocab_words,
        "word_text":    raw["word_text"],
        "word_pos":     raw["word_pos"],
        "trial_groups": raw["trial_groups"],
    }
    return compute_diagnostics(data, label="", k=k, verbose=False)


def avg_scalar_diagnostics(diag_list: List[Dict]) -> Dict:
    """Average scalar metrics across per-subject diagnostics dicts."""
    result: Dict = {}
    for key in SCALAR_METRICS:
        vals = []
        for d in diag_list:
            v = d.get(key, float("nan"))
            if isinstance(v, float) and not np.isnan(v):
                vals.append(v)
        result[key] = float(np.mean(vals)) if vals else float("nan")
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    sweep  = Path(args.sweep_dir)
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device    : {device}")
    print(f"Sweep dir : {sweep}")
    print(f"BERT      : {args.bert_name}  layer={args.bert_layer}")
    print(f"Preproc   : {PREPROC}")
    print()

    # ── Discover temperature directories ──────────────────────────────────────
    temp_dirs = sorted(d for d in sweep.iterdir()
                       if d.is_dir() and d.name.startswith("temp_"))
    temperatures = []
    for td in temp_dirs:
        num = td.name.replace("temp_", "").replace("_", ".")
        temperatures.append((float(num), td))
    print(f"Found {len(temperatures)} temperature configs: "
          f"{[t for t, _ in temperatures]}\n")

    # ── Load BERT hiddens once ────────────────────────────────────────────────
    print("Loading BERT hiddens ...")
    t0 = time.time()
    bert_hiddens = load_bert_hiddens(ONSET_DIR, args.bert_name,
                                     device="cpu", layer=args.bert_layer)
    print(f"  Done in {time.time()-t0:.1f}s\n")

    # Precompute fixed vocab BERT hiddens (raw, before bert_proj)
    h_vocab, vocab_words = build_h_vocab(bert_hiddens, args.poems)
    print(f"Vocab size: {len(vocab_words)} unique words\n")

    # ── Pre-load MEG data for ALL subjects (cached on CPU) ────────────────────
    meg_base = PREPROC if PREPROC.exists() else None
    if meg_base is None:
        print("[warn] icaed_Sai not found — falling back to raw .fif files (slow)\n")

    print("Pre-loading MEG data for all subjects ...")
    raw_cache: Dict[str, Optional[Dict]] = {}
    for sub in SUBJECTS:
        t0 = time.time()
        raw_cache[sub] = load_raw_subject(sub, args.sessions, args.poems,
                                          bert_hiddens, meg_base)
        n = len(raw_cache[sub]["meg"]) if raw_cache[sub] else 0
        print(f"  {sub}: {n:4d} windows  ({time.time()-t0:.1f}s)")
    print()

    # ── Main sweep ────────────────────────────────────────────────────────────
    entries = []
    total   = len(temperatures) * len(SUBJECTS)
    done    = 0

    for temp_val, temp_dir in temperatures:
        temp_tag = temp_dir.name      # e.g. "temp_0_1"
        base_dir = temp_dir / "inference" / args.model_tag

        for heldout in SUBJECTS:
            done += 1
            ckpt_dir = base_dir / f"loso_{heldout}"
            if not (ckpt_dir / "meg_encoder_best.pt").exists():
                print(f"  [{done}/{total}] T={temp_val} {heldout}: checkpoint missing — skip")
                continue

            # Load checkpoint weights
            enc  = MEGEncoder()
            proj = BERTTextProjection()
            s = torch.load(ckpt_dir / "meg_encoder_best.pt",
                           map_location="cpu", weights_only=False)
            if isinstance(s, dict) and "meg_encoder" in s:
                s = s["meg_encoder"]
            enc.load_state_dict(s)
            proj.load_state_dict(
                torch.load(ckpt_dir / "bert_proj_best.pt",
                           map_location="cpu", weights_only=False)
            )
            enc.eval().to(device)
            proj.eval().to(device)

            # Subjects for seen evaluation
            seen_subjects = [s for s in SUBJECTS if s != heldout]
            seen_single   = seen_subjects[0]  # first non-heldout alphabetically

            def _diag(sub: str) -> Optional[Dict]:
                raw = raw_cache.get(sub)
                if raw is None:
                    return None
                return run_diagnostics(raw, enc, proj, h_vocab, vocab_words,
                                       device, k=args.nn_k)

            # Unseen (heldout)
            diag_unseen = _diag(heldout)

            # Seen single
            diag_seen_single = _diag(seen_single)

            # Seen avg (all 12 non-heldout subjects)
            seen_diag_list = [d for sub in seen_subjects
                              if (d := _diag(sub)) is not None]
            diag_seen_avg = avg_scalar_diagnostics(seen_diag_list)
            seen_per_subject = {sub: _diag(sub) for sub in seen_subjects}

            entry = {
                "temperature":         temp_val,
                "temp_tag":            temp_tag,
                "heldout":             heldout,
                "ckpt_dir":            str(ckpt_dir),
                "seen_single_subject": seen_single,
                "unseen":              {k: diag_unseen[k] for k in SCALAR_METRICS}
                                       if diag_unseen else None,
                "seen_single":         {k: diag_seen_single[k] for k in SCALAR_METRICS}
                                       if diag_seen_single else None,
                "seen_avg":            diag_seen_avg,
                "seen_per_subject":    {sub: ({k: d[k] for k in SCALAR_METRICS}
                                              if d else None)
                                        for sub, d in seen_per_subject.items()},
            }
            entries.append(entry)

            n_unseen = len(raw_cache[heldout]["meg"]) if raw_cache[heldout] else 0
            ur1 = f"{diag_unseen['meg_effective_rank']:.2f}" if diag_unseen else "N/A"
            um  = f"{diag_unseen['nn_exact_match']:.3f}"     if diag_unseen else "N/A"
            print(f"  [{done}/{total}] T={temp_val}  {heldout}  "
                  f"unseen_erank={ur1}  unseen_NN={um}  (N={n_unseen})")

        # Free GPU memory between temperature groups
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out / "sweep_diagnostics.json"
    payload  = {
        "temperatures": [t for t, _ in temperatures],
        "subjects":     SUBJECTS,
        "metrics":      SCALAR_METRICS,
        "bert_name":    args.bert_name,
        "bert_layer":   args.bert_layer,
        "nn_k":         args.nn_k,
        "entries":      entries,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(entries)} entries → {out_path}")


if __name__ == "__main__":
    main()
