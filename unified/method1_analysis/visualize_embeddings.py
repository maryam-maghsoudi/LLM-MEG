#!/usr/bin/env python3
"""
visualize_embeddings.py — Method 1 MEG embedding collapse diagnostics.

Loads a trained MEGEncoder + BERTTextProjection from a LOSO checkpoint and
extracts z_meg / z_text for one seen and one unseen subject to diagnose
embedding collapse.

Outputs (all written to --out_dir):
    diagnostics.json      quantitative collapse stats per subject
    summary.md            one-page text summary (effective rank, cos sim, NN purity)
    umap_word.png         2D projection of z_meg, colored by word identity
    umap_wordpos.png      2D projection of z_meg, colored by word_pos (temporal)
    umap_joint.png        joint z_meg + z_text projection (circles vs X markers)
    heatmap.png           intra-trial pairwise cosine similarity heatmap

Run from llm_decoder/:
    python -m unified.method1_analysis.visualize_embeddings \\
        --ckpt_dir unified/out/inference/bert_base_uncased/loso_sub-01 \\
        --seen_subject sub-03 \\
        --device cuda
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_LLMDEC = Path(__file__).resolve().parent.parent.parent   # llm_decoder/
sys.path.insert(0, str(_LLMDEC))

from unified.data.base_dataset import MEGWordDataset, ONSET_DIR
from unified.data.splits import POEM_KEYS
from unified.methods.models import MEGEncoder, BERTTextProjection, load_bert_hiddens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize Method 1 MEG/text embeddings and diagnose collapse"
    )
    p.add_argument("--ckpt_dir", required=True,
                   help="LOSO checkpoint dir (meg_encoder_best.pt, "
                        "bert_proj_best.pt, run_config.json)")
    p.add_argument("--seen_subject", required=True,
                   help="Any training subject included in this checkpoint")
    p.add_argument("--unseen_subject", default=None,
                   help="LOSO heldout subject (default: inferred from run_config.json)")
    p.add_argument("--poem", nargs="+", default=None, choices=["poem1", "poem2"],
                   help="Poem(s) to include (default: both)")
    p.add_argument("--sessions", nargs="+", type=int, default=None,
                   help="Sessions to load per subject (default: 0-9)")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: <ckpt_dir>/embedding_vis/)")
    p.add_argument("--nn_purity_k", type=int, default=5,
                   help="k for NN purity (top-k most frequent words; default: 5)")
    return p.parse_args()


# ─── Checkpoint loading ───────────────────────────────────────────────────────

def load_checkpoint(ckpt_dir: Path, device: torch.device):
    """Return (meg_enc, bert_proj, bert_name, bert_layer, heldout_subject)."""
    ckpt_dir = Path(ckpt_dir)
    cfg: Dict = {}
    if (ckpt_dir / "run_config.json").exists():
        cfg = json.loads((ckpt_dir / "run_config.json").read_text())

    bert_name  = cfg.get("bert_name", "bert-base-uncased")
    bert_layer = int(cfg.get("bert_layer", -1))
    heldout    = cfg.get("heldout")

    enc = MEGEncoder()
    s = torch.load(ckpt_dir / "meg_encoder_best.pt", map_location="cpu",
                   weights_only=False)
    if isinstance(s, dict) and "meg_encoder" in s:
        s = s["meg_encoder"]
    enc.load_state_dict(s)
    enc.eval().to(device)

    proj = BERTTextProjection()
    proj.load_state_dict(
        torch.load(ckpt_dir / "bert_proj_best.pt", map_location="cpu",
                   weights_only=False)
    )
    proj.eval().to(device)

    print(f"Checkpoint : {ckpt_dir}")
    print(f"  bert={bert_name}  layer={bert_layer}  heldout={heldout}")
    return enc, proj, bert_name, bert_layer, heldout


# ─── Poem word lookup ─────────────────────────────────────────────────────────

def load_poem_vocab(poems: List[str]) -> Dict[str, Dict[int, str]]:
    """Return {poem: {word_pos: word_string}} from onset JSONs."""
    out: Dict[str, Dict[int, str]] = {}
    for poem in poems:
        entries = json.loads((ONSET_DIR / f"{poem}_word_onsets.json").read_text())
        out[poem] = {i: e["word"].strip().lower() for i, e in enumerate(entries)}
    return out


# ─── Embedding extraction ─────────────────────────────────────────────────────

def extract_embeddings(
    subject:      str,
    sessions:     List[int],
    poems:        List[str],
    meg_enc:      MEGEncoder,
    bert_proj:    BERTTextProjection,
    bert_hiddens: Dict[str, torch.Tensor],    # {poem: (N_words, 768)} on CPU
    poem_vocab:   Dict[str, Dict[int, str]],
    device:       torch.device,
    batch_size:   int = 256,
    meg_base:     Optional[Path] = None,
) -> Dict:
    """
    Run MEG windows and matched BERT hiddens through trained models.

    Returns
    -------
    z_meg        : Tensor(N, 128)  L2-normalized MEG embeddings
    z_text       : Tensor(N, 128)  projection of BERT hidden at same (poem, word_pos)
    z_text_vocab : Tensor(V, 128)  per-unique-word-type text embedding (mean-pooled BERT,
                                   like predict.py — used for NN purity)
    vocab_words  : List[str]       word string for each z_text_vocab row
    word_text    : List[str]       ground-truth word per MEG point
    word_pos     : List[int]       0-indexed position in poem
    poem         : List[str]
    session      : List[int]
    trial_id     : List[tuple]     (poem, session) per MEG point
    trial_groups : Dict[tuple, List[int]]  index lists into the above arrays
    """
    trials = [(subject, poem, sess) for poem in poems for sess in sessions]
    ds = MEGWordDataset(trials, augment=False, meg_base=meg_base)
    if len(ds) == 0:
        raise RuntimeError(
            f"No valid MEG windows for subject={subject}, "
            f"sessions={sessions}, poems={poems}"
        )

    n_trials = len(set(
        (ds._items[i]["poem"], ds._items[i]["session"])
        for i in range(len(ds._items))
    ))
    print(f"  {subject}: {len(ds)} valid word windows across {n_trials} trials")

    # ── batch inference ────────────────────────────────────────────────────────
    all_z_meg, all_z_text = [], []
    all_words, all_pos, all_poems, all_sessions, all_trials = [], [], [], [], []

    meg_enc.eval()
    bert_proj.eval()

    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            meg_batch, bert_batch = [], []
            for i in range(start, min(start + batch_size, len(ds))):
                item = ds[i]           # __getitem__ — returns torch tensors
                raw  = ds._items[i]    # raw dict for metadata
                meg_batch.append(item["meg_window"])
                bert_batch.append(bert_hiddens[raw["poem"]][raw["word_pos"]])
                all_words.append(raw["word_text"])
                all_pos.append(raw["word_pos"])
                all_poems.append(raw["poem"])
                all_sessions.append(raw["session"])
                all_trials.append((raw["poem"], raw["session"]))

            x = torch.stack(meg_batch).to(device)       # (B, 155, 40)
            h = torch.stack(bert_batch).to(device)      # (B, 768)
            all_z_meg.append(meg_enc(x).cpu())
            all_z_text.append(bert_proj(h).cpu())

    z_meg  = torch.cat(all_z_meg,  dim=0)    # (N, 128)
    z_text = torch.cat(all_z_text, dim=0)    # (N, 128)

    # ── trial index groups ─────────────────────────────────────────────────────
    trial_groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, t in enumerate(all_trials):
        trial_groups[t].append(i)

    # ── per-word-type vocab embeddings (mean-pool per unique word string) ──────
    # This matches predict.py: one embedding per unique word, mean over occurrences.
    vocab_vecs: List[torch.Tensor] = []
    vocab_words: List[str] = []
    with torch.no_grad():
        for poem in poems:
            h_poem = bert_hiddens[poem].float()   # (N_words, 768)
            pos_to_word = poem_vocab[poem]
            word_to_positions: Dict[str, List[int]] = defaultdict(list)
            for pos, word in pos_to_word.items():
                word_to_positions[word].append(pos)
            for word in sorted(word_to_positions):
                positions = word_to_positions[word]
                h_mean = h_poem[positions].mean(dim=0)               # (768,)
                z = bert_proj(h_mean.unsqueeze(0).to(device)).cpu()  # (1, 128)
                vocab_vecs.append(z)
                vocab_words.append(word)

    z_text_vocab = torch.cat(vocab_vecs, dim=0)   # (V, 128)

    return {
        "z_meg":        z_meg,
        "z_text":       z_text,
        "z_text_vocab": z_text_vocab,
        "vocab_words":  vocab_words,
        "word_text":    all_words,
        "word_pos":     all_pos,
        "poem":         all_poems,
        "session":      all_sessions,
        "trial_id":     all_trials,
        "trial_groups": dict(trial_groups),
    }


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def effective_rank(Z: torch.Tensor) -> float:
    """exp(entropy of normalised singular values) after mean-centering."""
    Z_c = Z.float() - Z.float().mean(dim=0)
    _, s, _ = torch.linalg.svd(Z_c, full_matrices=False)
    s = s.clamp(min=1e-12)
    p = s / s.sum()
    return float((-(p * p.log()).sum()).exp())


def pairwise_cos_stats(Z: torch.Tensor) -> Tuple[float, float]:
    """Mean and std of upper-triangle pairwise cosine similarities."""
    N = len(Z)
    if N < 2:
        return float("nan"), float("nan")
    Zn  = F.normalize(Z.float(), dim=-1)
    sim = Zn @ Zn.T
    idx = torch.triu_indices(N, N, offset=1)
    v   = sim[idx[0], idx[1]]
    return float(v.mean()), float(v.std())


def per_trial_cos_stats(data: Dict) -> Dict:
    """Aggregate intra-trial pairwise cosine stats across all trials."""
    means: List[float] = []
    for idxs in data["trial_groups"].values():
        if len(idxs) < 2:
            continue
        m, _ = pairwise_cos_stats(data["z_meg"][idxs])
        means.append(m)
    if not means:
        return {"mean": float("nan"), "std": float("nan"),
                "n_trials": 0, "per_trial": []}
    arr = np.array(means)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "n_trials": len(means), "per_trial": means}


def compute_nn_purity(data: Dict, k: int = 5) -> Dict:
    """
    For each z_meg point, find nearest z_text_vocab neighbour.
    Reports:
      - exact_match: fraction where NN word == true word
      - purity_top_k: fraction where NN word is in top-k most frequent words in poem
      - nn_word_distribution: top-10 predicted words by NN count
    """
    freq = Counter(data["word_text"])
    top_k_words = {w for w, _ in freq.most_common(k)}

    Zm = F.normalize(data["z_meg"].float(), dim=-1)
    Zv = F.normalize(data["z_text_vocab"].float(), dim=-1)
    sim    = Zm @ Zv.T                                    # (N, V)
    nn_idx = sim.argmax(dim=1).tolist()
    nn_words = [data["vocab_words"][i] for i in nn_idx]

    purity = sum(w in top_k_words for w in nn_words) / len(nn_words)
    exact  = sum(p == t for p, t in zip(nn_words, data["word_text"])) / len(nn_words)

    return {
        "purity_top_k": float(purity),
        "k": k,
        "exact_match":  float(exact),
        "top_k_words":  sorted(top_k_words),
        "nn_word_distribution": dict(Counter(nn_words).most_common(10)),
    }


def compute_diagnostics(data: Dict, label: str, k: int = 5, verbose: bool = True) -> Dict:
    if verbose:
        print(f"\n── Diagnostics: {label} ──")
    erank_meg  = effective_rank(data["z_meg"])
    erank_text = effective_rank(data["z_text"])
    cos_meg    = pairwise_cos_stats(data["z_meg"])
    cos_text   = pairwise_cos_stats(data["z_text"])
    trial_st   = per_trial_cos_stats(data)
    nn         = compute_nn_purity(data, k=k)

    if verbose:
        print(f"  MEG  effective_rank  = {erank_meg:.2f}  (max={data['z_meg'].shape[1]})")
        print(f"  text effective_rank  = {erank_text:.2f}")
        print(f"  MEG  pairwise cos    = {cos_meg[0]:.4f} ± {cos_meg[1]:.4f}")
        print(f"  text pairwise cos    = {cos_text[0]:.4f} ± {cos_text[1]:.4f}")
        print(f"  intra-trial MEG cos  = {trial_st['mean']:.4f} ± {trial_st['std']:.4f}"
              f"  ({trial_st['n_trials']} trials)")
        print(f"  NN exact match       = {nn['exact_match']:.3f}")
        print(f"  NN top-{k} purity    = {nn['purity_top_k']:.3f}"
              f"  (words: {nn['top_k_words']})")
        print(f"  NN distribution (top 10): {nn['nn_word_distribution']}")

    return {
        "label":                     label,
        "n_points":                  int(len(data["z_meg"])),
        "meg_effective_rank":        erank_meg,
        "text_effective_rank":       erank_text,
        "meg_pairwise_cos_mean":     cos_meg[0],
        "meg_pairwise_cos_std":      cos_meg[1],
        "text_pairwise_cos_mean":    cos_text[0],
        "text_pairwise_cos_std":     cos_text[1],
        "intra_trial_cos_mean":      trial_st["mean"],
        "intra_trial_cos_std":       trial_st["std"],
        "n_trials":                  trial_st["n_trials"],
        "nn_exact_match":            nn["exact_match"],
        f"nn_purity_top{k}":         nn["purity_top_k"],
        "nn_top_k_words":            nn["top_k_words"],
        "nn_word_distribution":      nn["nn_word_distribution"],
    }


# ─── Dimensionality reduction ─────────────────────────────────────────────────

def reduce_2d(X: np.ndarray, label: str = "") -> np.ndarray:
    """
    Fit UMAP if available, else t-SNE. Both use fixed random_state=42.
    Returns (N, 2) embedding for the same N rows of X.
    """
    X = X.astype(np.float32)
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(
            n_components=2, random_state=42,
            n_neighbors=min(15, len(X) - 1), min_dist=0.1,
        )
        print(f"  UMAP {label} ({len(X)} points) ...")
        return reducer.fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE
        print(f"  t-SNE {label} ({len(X)} points) ...")
        tsne = TSNE(
            n_components=2, random_state=42,
            perplexity=min(30, len(X) - 1), max_iter=1000,
        )
        return tsne.fit_transform(X)


# ─── Color helpers ────────────────────────────────────────────────────────────

_CMAP20 = plt.get_cmap("tab20")


def word_color_map(all_words: List[str], top_n: int = 20
                   ) -> Tuple[Dict[str, tuple], List[mpatches.Patch]]:
    """Return {word: rgba} dict and legend handles. Top top_n words get distinct colors."""
    freq = Counter(all_words)
    top_words = [w for w, _ in freq.most_common(top_n)]
    w2color: Dict[str, tuple] = {
        w: _CMAP20(i / 20) for i, w in enumerate(top_words)
    }
    handles = [mpatches.Patch(color=_CMAP20(i / 20), label=w)
               for i, w in enumerate(top_words)]
    handles.append(mpatches.Patch(color=(0.75, 0.75, 0.75), label="other"))
    return w2color, handles


def apply_colors(words: List[str], w2color: Dict[str, tuple]) -> List[tuple]:
    grey = (0.75, 0.75, 0.75, 0.5)
    return [w2color.get(w, grey) for w in words]


# ─── Plot functions ───────────────────────────────────────────────────────────

def _scatter(ax, xy, colors, marker="o", s=12, alpha=0.7, zorder=2):
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, marker=marker,
               s=s, alpha=alpha, linewidths=0, zorder=zorder)


def plot_umap_word(
    xy_seen:   np.ndarray, xy_unseen: np.ndarray,
    data_seen: Dict,       data_unseen: Dict,
    labels:    Tuple[str, str],
    out_path:  Path,
):
    """z_meg 2D projection colored by word identity."""
    # Build a shared color map over both subjects' words so colors are consistent
    all_words = data_seen["word_text"] + data_unseen["word_text"]
    w2color, handles = word_color_map(all_words)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, xy, data, lbl in zip(
        axes, [xy_seen, xy_unseen], [data_seen, data_unseen], labels
    ):
        colors = apply_colors(data["word_text"], w2color)
        _scatter(ax, xy, colors)
        ax.set_title(lbl, fontsize=13)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.legend(handles=handles, fontsize=6, ncol=2,
                  loc="upper right", framealpha=0.6, markerscale=1.5)
    fig.suptitle("z_meg 2D projection — colored by word identity\n"
                 "(collapse → all words cluster together instead of separating by label)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_umap_wordpos(
    xy_seen:   np.ndarray, xy_unseen: np.ndarray,
    data_seen: Dict,       data_unseen: Dict,
    labels:    Tuple[str, str],
    out_path:  Path,
):
    """z_meg 2D projection colored by word_pos (continuous, checks temporal structure)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, xy, data, lbl in zip(
        axes, [xy_seen, xy_unseen], [data_seen, data_unseen], labels
    ):
        sc = ax.scatter(xy[:, 0], xy[:, 1],
                        c=data["word_pos"], cmap="viridis",
                        s=12, alpha=0.7, linewidths=0)
        plt.colorbar(sc, ax=ax, label="word_pos")
        ax.set_title(lbl, fontsize=13)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.suptitle("z_meg 2D projection — colored by word position\n"
                 "(smooth gradient → temporal/positional structure dominates over word content)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_joint(
    xy_joint:     np.ndarray,
    n_seen:       int,
    n_unseen:     int,
    data_seen:    Dict,
    data_unseen:  Dict,
    vocab_words:  List[str],
    labels:       Tuple[str, str],
    out_path:     Path,
):
    """
    Joint 2D projection of z_meg (circles) and z_text_vocab (X markers).
    Embedding collapse is visible as MEG circles clustering near a small
    number of text anchors rather than their own word's anchor.
    """
    n_vocab = len(vocab_words)
    xy_meg_seen   = xy_joint[:n_seen]
    xy_meg_unseen = xy_joint[n_seen: n_seen + n_unseen]
    xy_text       = xy_joint[n_seen + n_unseen:]
    assert len(xy_text) == n_vocab

    all_words = data_seen["word_text"] + data_unseen["word_text"] + vocab_words
    w2color, handles = word_color_map(all_words)
    c_text = apply_colors(vocab_words, w2color)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, xy_meg, data, lbl in zip(
        axes,
        [xy_meg_seen, xy_meg_unseen],
        [data_seen, data_unseen],
        labels,
    ):
        c_meg = apply_colors(data["word_text"], w2color)
        # text anchors as large X markers (plotted first so MEG circles sit on top)
        ax.scatter(xy_text[:, 0], xy_text[:, 1],
                   c=c_text, marker="X", s=120, alpha=1.0,
                   linewidths=0.4, edgecolors="black", zorder=4)
        # MEG points as small circles
        ax.scatter(xy_meg[:, 0], xy_meg[:, 1],
                   c=c_meg, marker="o", s=14, alpha=0.55, linewidths=0, zorder=3)
        ax.set_title(lbl, fontsize=12)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.legend(handles=handles, fontsize=6, ncol=2,
                  loc="upper right", framealpha=0.6, markerscale=1.5)

    fig.suptitle("Joint z_meg ● and z_text ✕ projection — colored by word identity\n"
                 "Collapse: MEG circles pile near the same few ✕ markers regardless of true word",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def _pick_best_trial(data: Dict, min_points: int = 8) -> Optional[tuple]:
    """Return the trial key with the most valid MEG points."""
    best_key, best_n = None, -1
    for key, idxs in data["trial_groups"].items():
        if len(idxs) > best_n:
            best_key, best_n = key, len(idxs)
    return best_key if best_n >= min_points else None


def plot_heatmap(
    data_seen:   Dict,
    data_unseen: Dict,
    labels:      Tuple[str, str],
    out_path:    Path,
):
    """
    Pairwise cosine similarity heatmap for one representative trial per subject.
    A nearly uniform near-1 heatmap is the signature of a collapsed encoder.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data, lbl in zip(axes, [data_seen, data_unseen], labels):
        key = _pick_best_trial(data)
        if key is None:
            ax.text(0.5, 0.5, "No suitable trial (need ≥8 valid windows)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(lbl)
            continue

        idxs  = data["trial_groups"][key]
        Z     = F.normalize(data["z_meg"][idxs].float(), dim=-1)
        sim   = (Z @ Z.T).numpy()
        words = [data["word_text"][i] for i in idxs]
        N     = len(words)

        im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, label="cosine sim")
        fontsize = max(5, min(10, int(260 / N)))
        ax.set_xticks(range(N)); ax.set_xticklabels(words, rotation=90, fontsize=fontsize)
        ax.set_yticks(range(N)); ax.set_yticklabels(words, fontsize=fontsize)
        ax.set_title(f"{lbl}\ntrial: {key[0]} sess={key[1]}  (N={N} words)", fontsize=11)

    fig.suptitle("Intra-trial MEG pairwise cosine similarity\n"
                 "Uniform near-1 matrix → collapsed encoder (all z_meg point same direction)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ─── Markdown summary ─────────────────────────────────────────────────────────

def write_summary(
    stats_seen:   Dict,
    stats_unseen: Dict,
    out_path:     Path,
    k:            int,
):
    s, u = stats_seen, stats_unseen
    pk = f"nn_purity_top{k}"

    lines = [
        "# Method 1 Embedding Collapse Diagnostics",
        "",
        "| Metric | "
        f"Seen ({s['label']}) | Unseen ({u['label']}) |",
        "|---|---|---|",
        f"| N points | {s['n_points']} | {u['n_points']} |",
        f"| MEG effective rank | {s['meg_effective_rank']:.2f} | {u['meg_effective_rank']:.2f} |",
        f"| Text effective rank | {s['text_effective_rank']:.2f} | {u['text_effective_rank']:.2f} |",
        f"| MEG pairwise cos mean ± std | "
        f"{s['meg_pairwise_cos_mean']:.4f} ± {s['meg_pairwise_cos_std']:.4f} | "
        f"{u['meg_pairwise_cos_mean']:.4f} ± {u['meg_pairwise_cos_std']:.4f} |",
        f"| Text pairwise cos mean ± std | "
        f"{s['text_pairwise_cos_mean']:.4f} ± {s['text_pairwise_cos_std']:.4f} | "
        f"{u['text_pairwise_cos_mean']:.4f} ± {u['text_pairwise_cos_std']:.4f} |",
        f"| Intra-trial MEG cos mean ± std | "
        f"{s['intra_trial_cos_mean']:.4f} ± {s['intra_trial_cos_std']:.4f} | "
        f"{u['intra_trial_cos_mean']:.4f} ± {u['intra_trial_cos_std']:.4f} |",
        f"| NN exact match | {s['nn_exact_match']:.3f} | {u['nn_exact_match']:.3f} |",
        f"| NN top-{k} purity | {s[pk]:.3f} | {u[pk]:.3f} |",
        "",
        f"**Top-{k} words (seen):** {s['nn_top_k_words']}  ",
        f"**Top-{k} words (unseen):** {u['nn_top_k_words']}",
        "",
        "## NN predicted word distributions",
        f"**Seen** (top 10 NN predictions): {s['nn_word_distribution']}",
        "",
        f"**Unseen** (top 10 NN predictions): {u['nn_word_distribution']}",
        "",
        "## Interpretation guide",
        "- `MEG effective rank < 5` → near-total collapse (z_meg spans a tiny subspace)",
        "- `MEG pairwise cos > 0.9` → most vectors point the same direction",
        "- `Intra-trial MEG cos >> text cos` → collapse is specific to MEG encoder",
        "- `NN purity >> chance` (chance ≈ k/|vocab|) → predictions biased toward frequent words",
        "- `Seen >> Unseen` on exact match → cross-subject generalization failing",
    ]
    out_path.write_text("\n".join(lines))
    print(f"  Saved {out_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else ckpt_dir / "embedding_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    poems    = args.poem     or list(POEM_KEYS)
    sessions = args.sessions or list(range(10))

    # ── checkpoint ────────────────────────────────────────────────────────────
    meg_enc, bert_proj, bert_name, bert_layer, heldout = \
        load_checkpoint(ckpt_dir, device)

    unseen = args.unseen_subject or heldout
    if not unseen:
        raise ValueError(
            "Cannot determine unseen subject — pass --unseen_subject or "
            "ensure run_config.json contains a 'heldout' key"
        )
    seen = args.seen_subject
    if seen == unseen:
        raise ValueError(
            f"--seen_subject and --unseen_subject are the same ({seen}). "
            "Choose a different subject for one of them."
        )
    print(f"\nSeen: {seen}  |  Unseen: {unseen}")
    print(f"Poems: {poems}  Sessions: {sessions}")

    # ── BERT hiddens (CPU; shared across all subjects) ────────────────────────
    print("\nLoading BERT hiddens ...")
    bert_hiddens = load_bert_hiddens(ONSET_DIR, bert_name,
                                     device="cpu", layer=bert_layer)
    poem_vocab = load_poem_vocab(poems)

    # ── Extract embeddings ────────────────────────────────────────────────────
    print("\nExtracting embeddings ...")
    data_seen   = extract_embeddings(
        seen,   sessions, poems, meg_enc, bert_proj,
        bert_hiddens, poem_vocab, device,
    )
    data_unseen = extract_embeddings(
        unseen, sessions, poems, meg_enc, bert_proj,
        bert_hiddens, poem_vocab, device,
    )
    labels = (f"Seen: {seen}", f"Unseen: {unseen}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\nComputing diagnostics ...")
    stats_seen   = compute_diagnostics(data_seen,   seen,   k=args.nn_purity_k)
    stats_unseen = compute_diagnostics(data_unseen, unseen, k=args.nn_purity_k)

    diag_out = {"seen": stats_seen, "unseen": stats_unseen}
    (out_dir / "diagnostics.json").write_text(json.dumps(diag_out, indent=2))
    print("\n  Saved diagnostics.json")

    # ── Dimensionality reduction ───────────────────────────────────────────────
    print("\nFitting 2D reducers ...")
    n_seen   = len(data_seen["z_meg"])
    n_unseen = len(data_unseen["z_meg"])

    # z_meg projection (both subjects in the same 2D space)
    Z_meg = np.vstack([
        data_seen["z_meg"].float().numpy(),
        data_unseen["z_meg"].float().numpy(),
    ])
    xy_meg = reduce_2d(Z_meg, label="z_meg")
    xy_meg_seen   = xy_meg[:n_seen]
    xy_meg_unseen = xy_meg[n_seen:]

    # Joint projection (z_meg + z_text_vocab in the same 2D space)
    # z_text_vocab is the same for both subjects (derived from bert_hiddens + bert_proj)
    z_text_vocab = data_seen["z_text_vocab"]
    vocab_words  = data_seen["vocab_words"]
    Z_joint = np.vstack([
        data_seen["z_meg"].float().numpy(),
        data_unseen["z_meg"].float().numpy(),
        z_text_vocab.float().numpy(),
    ])
    xy_joint = reduce_2d(Z_joint, label="z_meg + z_text")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_umap_word(xy_meg_seen, xy_meg_unseen,
                   data_seen, data_unseen, labels,
                   out_dir / "umap_word.png")

    plot_umap_wordpos(xy_meg_seen, xy_meg_unseen,
                      data_seen, data_unseen, labels,
                      out_dir / "umap_wordpos.png")

    plot_joint(xy_joint, n_seen, n_unseen,
               data_seen, data_unseen, vocab_words, labels,
               out_dir / "umap_joint.png")

    plot_heatmap(data_seen, data_unseen, labels,
                 out_dir / "heatmap.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(stats_seen, stats_unseen,
                  out_dir / "summary.md", k=args.nn_purity_k)

    print(f"\nDone. Outputs → {out_dir}/")


if __name__ == "__main__":
    main()
