"""
probe_layers.py  —  Step 2
============================
Layer-selection probe: for each LLM layer, measure how well the cached hidden
states (hmid_t) separate word occurrences by word type via cosine k-NN retrieval.

Metric: mean Recall@1-5 (mR@1-5)
  For every word occurrence, rank all OTHER occurrences by cosine similarity.
  R@k = 1 if ANY of the top-k neighbors share the same word type (word string).
  Mean R@1-5 = average of R@1, R@2, R@3, R@4, R@5 — less noisy than R@1 alone.

Why k-NN (not linear probe)?
  With only ~117 word occurrences across 76 word types, a linear classifier
  is heavily underdetermined.  k-NN retrieval requires no training split and
  gives a direct measure of embedding cluster quality.

Outputs written to probe_out/<model_tag>/ (one subdirectory per model):
  probe_results.json      per-layer metrics (all R@k values + ranking)
  probe_plot.png          line plot: R@1..5 and mR@1-5 per layer
  probe_table.txt         human-readable ranked table (print to verify)

Usage
-----
  python probe_layers.py --llm_name HuggingFaceTB/SmolLM2-360M
  python probe_layers.py --llm_name gpt2
  python probe_layers.py --llm_name HuggingFaceTB/SmolLM2-1.7B
  python probe_layers.py --llm_name Qwen/Qwen2-0.5B
  python probe_layers.py --llm_name gpt2 --k 5      # mean over R@1..5 (default)
  python probe_layers.py --llm_name gpt2 --poems poem1
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
_HERE      = Path(__file__).parent
CACHE_ROOT = _HERE / "cache"      # model-specific subdirs: cache/<model_tag>/
OUT_ROOT   = _HERE / "probe_out"  # model-specific subdirs: probe_out/<model_tag>/
POEM_KEYS  = ["poem1", "poem2"]
LLM_NAME   = "HuggingFaceTB/SmolLM2-360M"


def model_tag(llm_name: str) -> str:
    return llm_name.replace("/", "_")


# ===========================================================================
#  LOAD CACHE
# ===========================================================================

def load_poem_cache(poem: str, cache_dir: Path) -> dict:
    path = cache_dir / f"{poem}_hiddens.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"Run cache_llm_hiddens.py first."
        )
    data = torch.load(path, map_location="cpu")
    print(f"  Loaded {poem}: {len(data['word_texts'])} words, "
          f"hidden_all_layers shape = {tuple(data['hidden_all_layers'].shape)}")
    return data


# ===========================================================================
#  K-NN RETRIEVAL PROBE
# ===========================================================================

def knn_recall(
    embeddings: torch.Tensor,   # (N, d)  L2-normalized
    word_types: list[str],       # (N,)  word string per occurrence
    max_k: int = 5,
) -> dict[int, float]:
    """
    Leave-one-out cosine k-NN retrieval.

    For each occurrence i, rank all j != i by cosine similarity.
    R@k[i] = 1 if any of the top-k neighbors share word_types[i].

    Returns {k: mean_recall_at_k for k in 1..max_k}
    """
    N = embeddings.shape[0]
    # Full cosine similarity matrix (N, N); diagonal is self-similarity = 1.0
    emb_norm = F.normalize(embeddings, dim=-1)
    sim = emb_norm @ emb_norm.T   # (N, N)

    # Set diagonal to -inf so each item never retrieves itself
    sim.fill_diagonal_(float("-inf"))

    # Sort descending; top_k_idx shape: (N, N-1)
    sorted_idx = sim.argsort(dim=-1, descending=True)

    recalls: dict[int, list[float]] = {k: [] for k in range(1, max_k + 1)}

    for i in range(N):
        true_type = word_types[i]
        neighbors = sorted_idx[i]          # sorted indices excluding self (due to -inf)

        hit_so_far = False
        for rank_0based, j in enumerate(neighbors[:max_k].tolist()):
            k = rank_0based + 1            # 1-indexed
            if word_types[j] == true_type:
                hit_so_far = True
            recalls[k].append(1.0 if hit_so_far else 0.0)

    return {k: float(np.mean(v)) for k, v in recalls.items()}


# ===========================================================================
#  SINGLE-LAYER PROBE
# ===========================================================================

def probe_one_layer(
    layer_idx: int,
    caches:    list[dict],
    max_k:     int = 5,
) -> dict:
    """
    Combine occurrences from all poems, extract hidden states for one layer,
    run k-NN probe, return recall metrics.
    """
    all_emb   = []
    all_words = []

    for cache in caches:
        # hidden_all_layers: (n_layers+1, N, d)
        emb = cache["hidden_all_layers"][layer_idx]   # (N, d)
        all_emb.append(emb)
        all_words.extend(cache["word_texts"])

    embeddings = torch.cat(all_emb, dim=0)   # (N_total, d)
    recalls    = knn_recall(embeddings, all_words, max_k=max_k)
    mean_r     = float(np.mean(list(recalls.values())))

    return {"layer": layer_idx, **{f"R@{k}": v for k, v in recalls.items()},
            f"mR@1-{max_k}": mean_r}


# ===========================================================================
#  FORMATTING
# ===========================================================================

def format_table(results: list[dict], max_k: int, n_layers: int) -> str:
    header = (
        f"{'Layer':>6}  {'Type':>12}  "
        + "  ".join(f"{'R@'+str(k):>6}" for k in range(1, max_k + 1))
        + f"  {'mR@1-'+str(max_k):>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        layer = r["layer"]
        label = (
            "embedding"  if layer == 0
            else f"block {layer:2d}" if layer <= n_layers
            else "?"
        )
        recall_cols = "  ".join(f"{r.get(f'R@{k}', 0):.4f}" for k in range(1, max_k + 1))
        mean_col    = f"{r[f'mR@1-{max_k}']:.4f}"
        lines.append(f"{layer:>6}  {label:>12}  {recall_cols}  {mean_col:>8}")

    return "\n".join(lines)


def format_ranked_table(results: list[dict], max_k: int, n_layers: int) -> str:
    ranked = sorted(results, key=lambda r: r[f"mR@1-{max_k}"], reverse=True)
    lines  = ["\n  Layers ranked by mR@1-5 (best first):"]
    lines.append(f"  {'Rank':>4}  {'Layer':>6}  {'Type':>12}  {'mR@1-'+str(max_k):>8}"
                 f"  {'R@1':>6}  {'R@5':>6}")
    lines.append("  " + "-" * 56)
    for rank, r in enumerate(ranked[:10], 1):
        layer = r["layer"]
        label = "embedding" if layer == 0 else f"block {layer}"
        lines.append(
            f"  {rank:>4}  {layer:>6}  {label:>12}  "
            f"{r[f'mR@1-{max_k}']:.4f}  {r.get('R@1',0):.4f}  {r.get(f'R@{max_k}',0):.4f}"
        )
    return "\n".join(lines)


# ===========================================================================
#  PLOT
# ===========================================================================

def save_plot(results: list[dict], max_k: int, n_layers: int, out_dir: Path) -> None:
    layers   = [r["layer"] for r in results]
    mean_r   = [r[f"mR@1-{max_k}"] for r in results]
    xticks   = layers
    xlabels  = [
        "emb" if l == 0 else str(l) for l in layers
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: individual R@k curves
    ax = axes[0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, max_k))
    for k, color in zip(range(1, max_k + 1), colors):
        vals = [r.get(f"R@{k}", 0) for r in results]
        ax.plot(layers, vals, marker="o", markersize=4, color=color, label=f"R@{k}")
    ax.set_ylabel("Recall@k")
    ax.set_title(f"Layer probe — cosine k-NN word-identity retrieval (n_layers={n_layers})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    # Mark best individual R@1
    best_r1_layer = max(results, key=lambda r: r.get("R@1", 0))["layer"]
    ax.axvline(best_r1_layer, color="steelblue", linestyle=":", alpha=0.6,
               label=f"best R@1 = layer {best_r1_layer}")

    # Bottom: mR@1-k
    ax2 = axes[1]
    ax2.plot(layers, mean_r, marker="s", color="darkred", linewidth=2,
             label=f"mR@1-{max_k}")
    best_mean_layer = max(results, key=lambda r: r[f"mR@1-{max_k}"])["layer"]
    ax2.axvline(best_mean_layer, color="darkred", linestyle="--", alpha=0.7,
                label=f"best mR@1-{max_k} = layer {best_mean_layer}")
    ax2.set_xlabel("Layer (0 = embedding, 1..n = transformer block)")
    ax2.set_ylabel(f"mR@1-{max_k}")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, fontsize=8)

    plt.tight_layout()
    path = out_dir / "probe_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ===========================================================================
#  MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Layer-selection probe for LLM hidden states")
    p.add_argument("--llm_name",  default=LLM_NAME,
                   help="HuggingFace model ID — used to locate cache/<model_tag>/")
    p.add_argument("--cache_dir", type=Path, default=None,
                   help="Override cache directory (default: cache/<model_tag>/)")
    p.add_argument("--out_dir",   type=Path, default=None,
                   help="Override output directory (default: probe_out/<model_tag>/)")
    p.add_argument("--poems",     nargs="+", default=POEM_KEYS)
    p.add_argument("--k",         type=int,  default=5,
                   help="Mean recall computed over R@1 .. R@k  (default 5)")
    return p.parse_args()


def main():
    args = parse_args()

    tag = model_tag(args.llm_name)
    if args.cache_dir is None:
        args.cache_dir = CACHE_ROOT / tag
    if args.out_dir is None:
        args.out_dir = OUT_ROOT / tag
    args.out_dir.mkdir(parents=True, exist_ok=True)
    max_k = args.k

    print(f"\n{'#'*60}")
    print(f"  STEP 2 — Layer selection probe")
    print(f"{'#'*60}")
    print(f"  Model     : {args.llm_name}")
    print(f"  Model tag : {tag}")
    print(f"  Cache dir : {args.cache_dir}")
    print(f"  Out dir   : {args.out_dir}")
    print(f"  Poems     : {args.poems}")
    print(f"  Metric    : mean R@1 .. R@{max_k}  (mR@1-{max_k})")

    # ── Load caches ───────────────────────────────────────────────────────────
    print(f"\nLoading caches ...")
    caches = [load_poem_cache(poem, args.cache_dir) for poem in args.poems]

    n_layers    = caches[0]["hidden_all_layers"].shape[0] - 1   # excl. embedding
    total_words = sum(len(c["word_texts"]) for c in caches)
    unique_words = len(set(w for c in caches for w in c["word_texts"]))

    print(f"\n  n_layers      : {n_layers}")
    print(f"  total word occurrences : {total_words}  ({' + '.join(str(len(c['word_texts'])) for c in caches)} across poems)")
    print(f"  unique word types      : {unique_words}")
    print(f"  layers to probe        : 0 .. {n_layers}  ({n_layers + 1} total)")

    # ── Probe every layer ────────────────────────────────────────────────────
    print(f"\nRunning k-NN probe on all {n_layers + 1} layers ...")
    results = []
    for layer_idx in range(n_layers + 1):
        label = "embedding" if layer_idx == 0 else f"block {layer_idx:2d}"
        r = probe_one_layer(layer_idx, caches, max_k=max_k)
        results.append(r)
        # Live progress line
        r1   = r.get("R@1", 0)
        r5   = r.get(f"R@{max_k}", 0)
        mean = r[f"mR@1-{max_k}"]
        print(f"    layer {layer_idx:2d} ({label:12s})  "
              f"R@1={r1:.4f}  R@{max_k}={r5:.4f}  mR@1-{max_k}={mean:.4f}")

    # ── Print full table ──────────────────────────────────────────────────────
    table_str = format_table(results, max_k, n_layers)
    print(f"\n  Full results table (by layer index):")
    print("  " + table_str.replace("\n", "\n  "))

    # ── Print ranked table ────────────────────────────────────────────────────
    ranked_str = format_ranked_table(results, max_k, n_layers)
    print(ranked_str)

    # ── Recommendation ───────────────────────────────────────────────────────
    best        = max(results, key=lambda r: r[f"mR@1-{max_k}"])
    best_layer  = best["layer"]
    best_label  = "embedding" if best_layer == 0 else f"block {best_layer}"
    chance_r1   = 1.0 / unique_words

    # Middle-third heuristic (for information)
    mid_lo  = round(n_layers / 3)
    mid_hi  = round(2 * n_layers / 3)
    best_mid = max(
        [r for r in results if mid_lo <= r["layer"] <= mid_hi],
        key=lambda r: r[f"mR@1-{max_k}"],
    )

    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION")
    print(f"{'='*60}")
    print(f"  Chance R@1 (random):       {chance_r1:.4f}")
    print(f"  Best overall  → layer {best_layer:2d} ({best_label})")
    print(f"    mR@1-{max_k} = {best[f'mR@1-{max_k}']:.4f}   "
          f"R@1 = {best.get('R@1',0):.4f}   R@{max_k} = {best.get(f'R@{max_k}',0):.4f}")
    print(f"  Best in middle third (layers {mid_lo}-{mid_hi}) → layer {best_mid['layer']}")
    print(f"    mR@1-{max_k} = {best_mid[f'mR@1-{max_k}']:.4f}")
    print(f"\n  Set HMID_LAYER = {best_layer} in your Stage 1 config.")
    print(f"  If you want to A/B: also try layer {best_mid['layer']} "
          f"(middle-third prior from Architecture.md).")

    # ── Save outputs ──────────────────────────────────────────────────────────
    probe_json = {
        "metric":    f"mR@1-{max_k}",
        "n_layers":  n_layers,
        "n_occurrences": total_words,
        "n_unique_word_types": unique_words,
        "chance_r1": chance_r1,
        "recommended_layer": best_layer,
        "middle_third_best_layer": best_mid["layer"],
        "results_by_layer": results,
        "results_ranked":   sorted(results, key=lambda r: r[f"mR@1-{max_k}"], reverse=True),
    }
    json_path = args.out_dir / "probe_results.json"
    json_path.write_text(json.dumps(probe_json, indent=2))
    print(f"\n  [saved] {json_path}")

    table_path = args.out_dir / "probe_table.txt"
    table_path.write_text(table_str + "\n" + ranked_str)
    print(f"  [saved] {table_path}")

    save_plot(results, max_k, n_layers, args.out_dir)

    print(f"\n{'='*60}")
    print(f"  STEP 2 COMPLETE  [{args.llm_name}]")
    print(f"{'='*60}")
    print(f"  Outputs in {args.out_dir}/")
    print(f"    probe_results.json  — full metrics")
    print(f"    probe_table.txt     — human-readable table")
    print(f"    probe_plot.png      — R@1-{max_k} and mR@1-{max_k} per layer")
    print(f"\n  Next: set HMID_LAYER = {best_layer} and run Stage 1 training.")


if __name__ == "__main__":
    main()
