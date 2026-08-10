"""
cache_llm_hiddens.py  —  Step 1
================================
Run a frozen causal LLM once over both poem texts and cache, per word occurrence:

  hmid_t : hidden state at EVERY layer, mean-pooled over the word's subword tokens
           shape per poem: (n_layers+1, N_words, d_model)
           layer 0 = token embeddings, layers 1..n = after each transformer block

  lm_hid_final : final-layer hidden state at the LAST subword position of each word
                 shape: (N_words, d_model)  — used for p_t in Stage 2

  lm_logits_restricted : lm_head(lm_hid_final) restricted to the 76-word closed vocab
                         shape: (N_words, R)  where R = restricted vocab size

Outputs written to cache/<model_tag>/ (one subdirectory per model):
  poem1_hiddens.pt           tensor cache for poem 1
  poem2_hiddens.pt           tensor cache for poem 2
  vocab_info.json            restricted vocabulary + LLM metadata
  alignment_poem1.txt        human-readable word→token alignment (verify this!)
  alignment_poem2.txt
  cache_summary.json         shapes + stats for sanity checking

Model tag is derived from the HuggingFace model ID by replacing "/" with "_",
e.g. "HuggingFaceTB/SmolLM2-360M" → cache/HuggingFaceTB_SmolLM2-360M/

Usage
-----
  python cache_llm_hiddens.py                                  # SmolLM2-360M (default)
  python cache_llm_hiddens.py --llm_name gpt2
  python cache_llm_hiddens.py --llm_name HuggingFaceTB/SmolLM2-1.7B
  python cache_llm_hiddens.py --llm_name Qwen/Qwen2-0.5B
  python cache_llm_hiddens.py --poem poem1                     # single poem only
"""

import argparse
import json
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
#  Paths  (relative to this file's location)
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).parent
ONSET_DIR  = _HERE.parent.parent / "contrastive_learning" / "onset_out"
CACHE_ROOT = _HERE / "cache"          # model-specific subdirs live here: cache/<model_tag>/
# Recommended models (spec: SmolLM or Qwen).
# GPT-2 is a quick sanity-check fallback (already cached locally).
#
#   HuggingFaceTB/SmolLM2-360M    ~720 MB   recommended starting point
#   HuggingFaceTB/SmolLM2-1.7B    ~3.4 GB   strongest SmolLM
#   Qwen/Qwen2-0.5B               ~1.0 GB   Qwen2 entry point
#   gpt2                           ~500 MB   baseline / fallback
LLM_NAME   = "HuggingFaceTB/SmolLM2-360M"
POEM_KEYS  = ["poem1", "poem2"]


def model_tag(llm_name: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe directory name.
    e.g. 'HuggingFaceTB/SmolLM2-360M' -> 'HuggingFaceTB_SmolLM2-360M'
    """
    return llm_name.replace("/", "_")


# ===========================================================================
#  WORD-TOKEN ALIGNMENT
# ===========================================================================

def build_alignment(words: list[str], tokenizer) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """
    Tokenize each word with a leading space (mid-sentence BPE convention),
    concatenate into one token sequence, and record each word's token span.

    Returns
    -------
    input_ids : (T_total,) long tensor
    spans     : list of (start, end_exclusive) index pairs, one per word
    """
    all_ids: list[int] = []
    spans:   list[tuple[int, int]] = []

    for word in words:
        ids = tokenizer.encode(" " + word.strip().lower(), add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(word.strip().lower(), add_special_tokens=False)
        if not ids:
            ids = [tokenizer.unk_token_id or 0]

        start = len(all_ids)
        all_ids.extend(ids)
        spans.append((start, len(all_ids)))

    return torch.tensor(all_ids, dtype=torch.long), spans


def format_alignment_table(words, spans, tokenizer, input_ids) -> str:
    """Return a human-readable alignment table for one poem."""
    lines = [
        f"{'#':>4}  {'Word':>15}  {'Span':>12}  {'Tokens'}",
        "-" * 72,
    ]
    for i, (word, (s, e)) in enumerate(zip(words, spans)):
        tok_ids   = input_ids[s:e].tolist()
        tok_strs  = [repr(tokenizer.decode([t])) for t in tok_ids]
        span_str  = f"[{s}:{e}]"
        lines.append(f"{i:>4}  {word:>15}  {span_str:>12}  {' '.join(tok_strs)}")
    lines.append(f"\nTotal words: {len(words)}   Total tokens: {len(input_ids)}")
    return "\n".join(lines)


# ===========================================================================
#  RESTRICTED VOCABULARY
# ===========================================================================

def build_restricted_vocab(onset_dir: Path, poem_keys: list[str], tokenizer) -> dict:
    """
    Collect all unique words across both poems.
    For each word, record the LLM token ID of its first subword (with leading space).
    Multi-token words are represented by their first-token probability at inference.

    Returns a dict with keys: words, first_token_ids, multi_token_flags
    """
    all_words: list[str] = []
    seen = set()
    for poem in poem_keys:
        path = onset_dir / f"{poem}_word_onsets.json"
        onsets = json.loads(path.read_text())
        for entry in onsets:
            w = entry["word"].strip().lower()
            if w not in seen:
                seen.add(w)
                all_words.append(w)

    all_words = sorted(all_words)   # deterministic ordering

    first_token_ids   = []
    multi_token_flags = []
    for w in all_words:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(w, add_special_tokens=False)
        first_token_ids.append(ids[0] if ids else tokenizer.unk_token_id)
        multi_token_flags.append(len(ids) > 1)

    n_multi = sum(multi_token_flags)
    print(f"\n[vocab]  {len(all_words)} unique words across {poem_keys}")
    print(f"         {n_multi} words are multi-token (first-token approximation for p_t)")
    if n_multi:
        multi_words = [w for w, f in zip(all_words, multi_token_flags) if f]
        print(f"         multi-token words: {multi_words}")

    return {
        "words":             all_words,
        "first_token_ids":   first_token_ids,
        "multi_token_flags": multi_token_flags,
    }


# ===========================================================================
#  LLM FORWARD PASS
# ===========================================================================

@torch.no_grad()
def run_llm(model, input_ids: torch.Tensor, device: torch.device):
    """
    Single forward pass; returns all hidden states and logits.

    Returns
    -------
    hidden_states : tuple of (1, T, d) tensors, length n_layers+1
                   index 0 = token embeddings, 1..n = after each transformer block
    logits        : (1, T, vocab_size)
    """
    ids = input_ids.unsqueeze(0).to(device)
    out = model(ids, output_hidden_states=True)
    return out.hidden_states, out.logits


# ===========================================================================
#  PER-POEM CACHING
# ===========================================================================

def cache_poem(
    poem:           str,
    onset_dir:      Path,
    tokenizer,
    model,
    device:         torch.device,
    restricted_vocab: dict,
    cache_dir:      Path,
) -> dict:
    """
    Process one poem: build alignment, run LLM, pool per word, save .pt.

    Saved tensor dict keys
    ----------------------
    word_texts          : list[str]              length N
    word_spans          : list[(int,int)]        token spans, length N
    hidden_all_layers   : (n_layers+1, N, d)    mean-pooled over subword span
    lm_hid_final        : (N, d)                final layer, last-subword position
    lm_logits_restricted: (N, R)                logits over restricted vocab

    Returns a summary dict for cache_summary.json.
    """
    print(f"\n{'='*60}")
    print(f"  Processing {poem}")
    print(f"{'='*60}")

    # ── Load onsets ──────────────────────────────────────────────────────────
    onsets = json.loads((onset_dir / f"{poem}_word_onsets.json").read_text())
    words  = [e["word"].strip().lower() for e in onsets]
    N      = len(words)
    print(f"  Words in poem: {N}")
    print(f"  First 5 words: {words[:5]}")

    # ── Build alignment ───────────────────────────────────────────────────────
    input_ids, spans = build_alignment(words, tokenizer)
    T_total = len(input_ids)
    print(f"  Token sequence length: {T_total}")

    # Save human-readable alignment table
    alignment_txt = format_alignment_table(words, spans, tokenizer, input_ids)
    aln_path = cache_dir / f"alignment_{poem}.txt"
    aln_path.write_text(alignment_txt)
    print(f"  [saved] alignment table → {aln_path}")
    print("\n  --- Alignment (first 10 words) ---")
    for line in alignment_txt.split("\n")[:13]:
        print("  " + line)
    print("  ...")

    # ── Run LLM ───────────────────────────────────────────────────────────────
    print(f"\n  Running LLM forward pass ({model.__class__.__name__}) ...")
    hidden_states, logits = run_llm(model, input_ids, device)

    n_layers = len(hidden_states) - 1
    d_model  = hidden_states[0].shape[-1]
    print(f"  n_layers={n_layers}  d_model={d_model}  "
          f"hidden_states tuple length={len(hidden_states)}")

    # ── Pool per word ─────────────────────────────────────────────────────────
    # hidden_all_layers[l, i, :] = mean of hidden_states[l][0, s:e, :]
    # lm_hid_final[i, :]         = hidden_states[-1][0, e-1, :]  (last subword)

    hidden_all  = torch.zeros(n_layers + 1, N, d_model)
    lm_hid_fin  = torch.zeros(N, d_model)

    for i, (s, e) in enumerate(spans):
        for l, hs in enumerate(hidden_states):
            hidden_all[l, i, :] = hs[0, s:e, :].mean(dim=0).cpu()
        lm_hid_fin[i, :] = hidden_states[-1][0, e - 1, :].cpu()

    print(f"  hidden_all_layers shape : {tuple(hidden_all.shape)}")
    print(f"  lm_hid_final shape      : {tuple(lm_hid_fin.shape)}")

    # ── Restricted vocab logits ───────────────────────────────────────────────
    # lm_head(lm_hid_final) restricted to the R vocab first-token IDs
    lm_head = model.lm_head
    R        = len(restricted_vocab["words"])
    r_ids    = torch.tensor(restricted_vocab["first_token_ids"], dtype=torch.long)

    lm_hid_fin_dev = lm_hid_fin.to(device)
    full_logits    = lm_head(lm_hid_fin_dev)            # (N, vocab_size)
    restricted_log = full_logits[:, r_ids].cpu()        # (N, R)

    print(f"  restricted vocab size   : {R}")
    print(f"  lm_logits_restricted    : {tuple(restricted_log.shape)}")

    # Quick sanity check: argmax restricted logit for first few words
    print("\n  [sanity] restricted-vocab argmax for first 5 word positions:")
    rv_words = restricted_vocab["words"]
    for i in range(min(5, N)):
        pred_idx  = restricted_log[i].argmax().item()
        pred_word = rv_words[pred_idx]
        true_word = words[i]
        print(f"    pos {i:2d}  true='{true_word}'  top-1 predicted='{pred_word}'")

    # ── Save ─────────────────────────────────────────────────────────────────
    cache_data = {
        "word_texts":            words,
        "word_spans":            spans,
        "hidden_all_layers":     hidden_all,       # (n_layers+1, N, d)
        "lm_hid_final":          lm_hid_fin,       # (N, d)
        "lm_logits_restricted":  restricted_log,   # (N, R)
    }
    out_path = cache_dir / f"{poem}_hiddens.pt"
    torch.save(cache_data, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  [saved] {out_path}  ({size_mb:.1f} MB)")

    return {
        "poem":              poem,
        "n_words":           N,
        "n_tokens":          T_total,
        "n_layers":          n_layers,
        "d_model":           d_model,
        "hidden_shape":      list(hidden_all.shape),
        "restricted_size":   R,
        "file_mb":           round(size_mb, 2),
    }


# ===========================================================================
#  MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Cache LLM hidden states for poem stimuli")
    p.add_argument("--llm_name",   default=LLM_NAME,
                   help="HuggingFace model ID (default: SmolLM2-360M)")
    p.add_argument("--poems",      nargs="+", default=POEM_KEYS,
                   choices=POEM_KEYS, help="Which poems to process")
    p.add_argument("--cache_dir",  type=Path, default=None,
                   help="Override cache directory (default: cache/<model_tag>/)")
    p.add_argument("--onset_dir",  type=Path, default=ONSET_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    # Default cache dir is namespaced by model so multiple models coexist
    if args.cache_dir is None:
        args.cache_dir = CACHE_ROOT / model_tag(args.llm_name)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*60}")
    print(f"  STEP 1 — LLM hidden state caching")
    print(f"{'#'*60}")
    print(f"  LLM        : {args.llm_name}")
    print(f"  Model tag  : {model_tag(args.llm_name)}")
    print(f"  Poems      : {args.poems}")
    print(f"  Cache dir  : {args.cache_dir}")
    print(f"  Device     : {device}")

    # ── Load tokenizer + model ────────────────────────────────────────────────
    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.llm_name} ...")
    model = AutoModelForCausalLM.from_pretrained(args.llm_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters (frozen, eval mode)")

    # ── Build restricted vocabulary (shared across poems) ─────────────────────
    restricted_vocab = build_restricted_vocab(args.onset_dir, POEM_KEYS, tokenizer)

    # ── Cache each poem ───────────────────────────────────────────────────────
    summaries = []
    for poem in args.poems:
        summary = cache_poem(
            poem         = poem,
            onset_dir    = args.onset_dir,
            tokenizer    = tokenizer,
            model        = model,
            device       = device,
            restricted_vocab = restricted_vocab,
            cache_dir    = args.cache_dir,
        )
        summaries.append(summary)

    # ── Save vocab info + summary ─────────────────────────────────────────────
    vocab_info = {
        "llm_name":          args.llm_name,
        "n_layers":          summaries[0]["n_layers"],
        "d_model":           summaries[0]["d_model"],
        "restricted_words":       restricted_vocab["words"],
        "restricted_first_token_ids": restricted_vocab["first_token_ids"],
        "multi_token_flags":      restricted_vocab["multi_token_flags"],
    }
    vocab_path = args.cache_dir / "vocab_info.json"
    vocab_path.write_text(json.dumps(vocab_info, indent=2))
    print(f"\n[saved] vocab_info.json  ({len(restricted_vocab['words'])} words)")

    cache_summary = {
        "llm_name": args.llm_name,
        "poems":    summaries,
    }
    summary_path = args.cache_dir / "cache_summary.json"
    summary_path.write_text(json.dumps(cache_summary, indent=2))

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 1 COMPLETE")
    print(f"{'='*60}")
    for s in summaries:
        print(f"  {s['poem']:8s}  words={s['n_words']:3d}  tokens={s['n_tokens']:3d}  "
              f"hidden_shape={s['hidden_shape']}  {s['file_mb']} MB")
    print(f"\n  Files in {args.cache_dir}:")
    for f in sorted(args.cache_dir.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size / 1e3:.0f} KB)")
    print(f"\n  Run probe_layers.py next to select the best LLM layer.")


if __name__ == "__main__":
    main()
