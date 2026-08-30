"""
predict.py — inference API for all three methods.

    result = predict(
        subject   = "sub-01",
        session   = 3,
        condition = "lis",
        poem      = "poem1",
        method    = "twostage",
        ckpt_dir  = "out/twostage/.../",
        device    = "cuda",
    )

result keys
-----------
words      : List[str]            ground-truth word sequence for this trial
pred_top1  : List[str]            argmax prediction at each position
scores     : Tensor(N, |V|)       raw scores (logits/cosines) over eval vocab
vocab      : List[str]            evaluation vocabulary (sorted unique words in test split)
valid      : List[bool]           True where MEG window was available
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Module-level cache: keyed by (bert_name, layer) → {poem: Tensor}
_bert_hiddens_cache: Dict = {}

from unified.data.base_dataset import (
    _load_meg_trial, _load_onsets, _onset_to_window,
    N_CHANNELS, WIN_SIZE,
)
from unified.methods.models import (
    MEGEncoder, BERTTextProjection, LLMTextProjection,
    GRUHead, Adapter, load_bert_hiddens, load_lm_head,
)

_HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _load_run_config(ckpt_dir: Path) -> Dict:
    cfg_path = ckpt_dir / "run_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def _load_meg_windows(
    subject: str, session: int, condition: str, poem: str
) -> tuple:
    """
    Returns (windows, valid_mask, word_texts) for one trial.
    windows    : (N, N_CHANNELS, WIN_SIZE) float32 tensor
    valid_mask : (N,) bool tensor
    word_texts : List[str]
    """
    onset_list = _load_onsets(poem)
    N          = len(onset_list)
    data       = _load_meg_trial(subject, f"{poem}{condition}", session)

    import numpy as np
    windows    = torch.zeros(N, N_CHANNELS, WIN_SIZE)
    valid_mask = torch.zeros(N, dtype=torch.bool)
    word_texts = [e["word"].strip().lower() for e in onset_list]

    if data is not None:
        n_t = data.shape[1]
        for i, entry in enumerate(onset_list):
            idx = _onset_to_window(entry["start"], n_t)
            if idx is None:
                continue
            s, e = idx
            win  = data[:, s:e]
            if win.shape[-1] != WIN_SIZE:
                continue
            windows[i]    = torch.from_numpy(win)
            valid_mask[i] = True

    return windows, valid_mask, word_texts


def _build_vocab(word_texts: List[str]) -> List[str]:
    """Sorted unique words in the sequence."""
    return sorted(set(word_texts))


# ---------------------------------------------------------------------------
#  Per-method predict functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_inference(
    windows:    torch.Tensor,
    valid_mask: torch.Tensor,
    word_texts: List[str],
    ckpt_dir:   Path,
    device:     torch.device,
    cfg:        Dict,
) -> Dict:
    """
    Method 1: cosine similarity between MEG embedding and BERT text embeddings.
    scores[i, j] = cosine_sim(z_meg_i, z_text_vocab[j])
    """
    bert_name = cfg.get("bert_name", "bert-base-uncased")
    bert_layer= int(cfg.get("bert_layer", -1))

    # Load models
    meg_enc = MEGEncoder().to(device)
    meg_enc.load_state_dict(
        torch.load(ckpt_dir / "meg_encoder_best.pt", map_location="cpu")
    )
    meg_enc.eval()

    bert_proj = BERTTextProjection().to(device)
    bert_proj.load_state_dict(
        torch.load(ckpt_dir / "bert_proj_best.pt", map_location="cpu")
    )
    bert_proj.eval()

    # BERT hidden states for all words in the poem — cached across trials
    from unified.data.base_dataset import ONSET_DIR
    poem = cfg.get("_poem", word_texts)                    # fallback
    _cache_key = (bert_name, bert_layer)
    if _cache_key not in _bert_hiddens_cache:
        _bert_hiddens_cache[_cache_key] = load_bert_hiddens(ONSET_DIR, bert_name, str(device), bert_layer)
    bert_hiddens = _bert_hiddens_cache[_cache_key]

    # Build vocab from this trial
    vocab    = _build_vocab(word_texts)
    vocab_set= {w: i for i, w in enumerate(vocab)}

    # Encode vocab words using their BERT hidden states (we need the poem)
    # For predict, we use the mean BERT hidden over all occurrences of each vocab word
    # (they have different positions, hence different contextual representations)
    # We aggregate by taking the mean over all occurrences.
    from collections import defaultdict
    poem_key = cfg.get("poem", "poem1")
    h_all    = bert_hiddens[poem_key]           # (N_words, 768)
    onset_list = _load_onsets(poem_key)
    word_list  = [e["word"].strip().lower() for e in onset_list]

    vocab_hiddens: Dict[str, List] = defaultdict(list)
    for pos, w in enumerate(word_list):
        if w in vocab_set:
            vocab_hiddens[w].append(h_all[pos])

    # Average occurrences → (|V|, 768), project → (|V|, 128)
    vocab_h = torch.stack([
        torch.stack(vocab_hiddens[w]).mean(0) for w in vocab
    ]).to(device)                          # (|V|, 768)
    z_vocab = bert_proj(vocab_h)           # (|V|, 128)

    # Encode MEG windows
    N      = windows.shape[0]
    scores = torch.zeros(N, len(vocab))
    x      = windows.to(device)           # (N, C, T)
    z_meg  = meg_enc(x)                   # (N, 128)

    # cosine similarity (both already L2-normalised)
    scores = (z_meg @ z_vocab.T).cpu()    # (N, |V|)

    return scores, vocab


@torch.no_grad()
def _predict_twostage(
    windows:    torch.Tensor,
    valid_mask: torch.Tensor,
    word_texts: List[str],
    ckpt_dir:   Path,
    device:     torch.device,
    cfg:        Dict,
) -> tuple:
    """
    Method 2: GRU next-word scores restricted to eval vocab.
    Falls back to Stage 1 cosine similarity if stage2_best.pt is absent.
    """
    from unified.methods.train_twostage import MODEL_CONFIGS, _model_tag, _load_cache

    llm_name   = cfg.get("llm_name", "HuggingFaceTB/SmolLM2-360M")
    d_model    = MODEL_CONFIGS[llm_name]["d_model"]
    gru_hidden = int(cfg.get("gru_hidden", 256))
    cache_root = Path(cfg.get("cache_root",
                    str(_HERE.parent / "llm_twostage" / "cache")))

    vocab    = _build_vocab(word_texts)
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    # MEGEncoder
    s1_ckpt  = torch.load(ckpt_dir / "stage1_best.pt", map_location="cpu")
    meg_enc  = MEGEncoder().to(device)
    meg_enc.load_state_dict(s1_ckpt["meg_encoder"])
    meg_enc.eval()

    N   = windows.shape[0]
    x   = windows.to(device)
    z   = meg_enc(x).unsqueeze(0)    # (1, N, 128)

    stage2_path = ckpt_dir / "stage2_best.pt"
    if stage2_path.exists():
        # Stage 2 scores via GRU + lm_head restricted to vocab
        s2_ckpt  = torch.load(stage2_path, map_location="cpu")
        gru_head = GRUHead(gru_hidden=gru_hidden, d_model=d_model).to(device)
        gru_head.load_state_dict(s2_ckpt["gru_head"])
        gru_head.eval()

        lm_head = load_lm_head(llm_name, device)

        # vocab word → first-token ID in this LLM
        vocab_info = json.loads(
            (cache_root / _model_tag(llm_name) / "vocab_info.json").read_text()
        )
        word2tokid = dict(zip(vocab_info["restricted_words"],
                              vocab_info["restricted_first_token_ids"]))
        tokenizer  = AutoTokenizer.from_pretrained(llm_name)
        # Get first-token IDs for eval vocab words
        r_ids = []
        for w in vocab:
            if w in word2tokid:
                r_ids.append(word2tokid[w])
            else:
                ids = tokenizer.encode(" " + w, add_special_tokens=False)
                r_ids.append(ids[0] if ids else tokenizer.unk_token_id)
        r_ids_t = torch.tensor(r_ids, dtype=torch.long, device=device)

        # Apply the same control that was used during training so that
        # shuffle_time / zero models are evaluated under matching conditions.
        control = cfg.get("control", "none")
        if control == "zero":
            z = torch.zeros_like(z)
        elif control == "shuffle_time":
            perm = torch.randperm(z.size(1), device=z.device)
            z = z[:, perm, :]

        y         = gru_head(z)                     # (1, N, d_model)
        all_logits= lm_head(y)[0]                   # (N, vocab_size)
        raw       = all_logits[:, r_ids_t].cpu()    # (N, |V|)

        # raw[i] = lm_head(y_i) predicts word i+1, not word i.
        # Shift so that scores[i] is the prediction for word i (from context z_0..z_{i-1}).
        # Position 0 has no GRU predecessor → leave as zeros (evaluated as random rank).
        scores = torch.zeros_like(raw)
        scores[1:] = raw[:-1]

    else:
        # Fallback: Stage 1 cosine similarity against LLM text hiddens
        hmid_layer = int(cfg.get("hmid_layer_used",
                                  MODEL_CONFIGS[llm_name]["hmid_layer"]))
        poem_key   = cfg.get("poem", "poem1")
        cache      = _load_cache(llm_name, poem_key, cache_root)
        h_all      = cache["hidden_all_layers"][hmid_layer]  # (N_words, d_model)

        text_proj  = LLMTextProjection(d_model=d_model).to(device)
        text_proj.load_state_dict(s1_ckpt["text_proj"])
        text_proj.eval()

        onset_list = _load_onsets(poem_key)
        word_list  = [e["word"].strip().lower() for e in onset_list]

        from collections import defaultdict
        vocab_hiddens: Dict[str, list] = defaultdict(list)
        for pos, w in enumerate(word_list):
            if w in {v: 0 for v in vocab}:
                vocab_hiddens[w].append(h_all[pos])

        vocab_h = torch.stack([
            torch.stack(vocab_hiddens[w]).mean(0) for w in vocab
        ]).to(device)
        z_vocab  = text_proj(vocab_h)      # (|V|, 128)
        scores   = (z[0] @ z_vocab.T).cpu()  # (N, |V|)

    return scores, vocab


@torch.no_grad()
def _predict_interleaved(
    windows:    torch.Tensor,
    valid_mask: torch.Tensor,
    word_texts: List[str],
    ckpt_dir:   Path,
    device:     torch.device,
    cfg:        Dict,
) -> tuple:
    """
    Method 3: soft-token-conditioned LLM next-token logits restricted to eval vocab.
    Teacher-forced: each word's soft token sees the ground-truth preceding context.
    """
    llm_name = cfg.get("llm_name", "gpt2")
    n_soft   = int(cfg.get("n_soft", 1))

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    token_emb = llm.get_input_embeddings()
    d_model   = token_emb.embedding_dim

    meg_enc_ckpt = cfg.get("meg_enc_ckpt")
    meg_enc = MEGEncoder().to(device)
    meg_enc.load_state_dict(
        torch.load(meg_enc_ckpt, map_location="cpu") if meg_enc_ckpt
        else torch.load(ckpt_dir / "meg_encoder_best.pt", map_location="cpu")
    )
    meg_enc.eval()

    adapter = Adapter(n_soft=n_soft, d_model=d_model).to(device)
    adapter.load_state_dict(
        torch.load(ckpt_dir / "adapter_best.pt", map_location="cpu")
    )
    adapter.eval()

    # Tokenise word_texts
    token_ids_per_word = []
    for w in word_texts:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if not ids:
            ids = [tokenizer.unk_token_id or 0]
        token_ids_per_word.append(ids)

    vocab      = _build_vocab(word_texts)
    vocab_ids  = []
    for w in vocab:
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        vocab_ids.append(ids[0] if ids else tokenizer.unk_token_id)
    vocab_ids_t = torch.tensor(vocab_ids, dtype=torch.long, device=device)

    N      = len(word_texts)
    scores = torch.zeros(N, len(vocab))

    # Build interleaved sequence, collect the logit at each word boundary
    from unified.methods.train_interleaved import _build_interleaved
    embeds, _ = _build_interleaved(
        windows, valid_mask, token_ids_per_word,
        meg_enc, adapter, token_emb, device, n_soft,
    )
    out = llm(inputs_embeds=embeds)
    logits = out.logits[0]                # (L, vocab_size)

    # The position just before each word's first text token is where the LLM
    # predicts that word. Soft positions: n_soft per word, then text tokens.
    pos = 0
    for i, tok_ids in enumerate(token_ids_per_word):
        # pos = start of this word's soft token(s)
        # prediction position = last soft token (just before first text token)
        pred_pos = pos + n_soft - 1
        if pred_pos < logits.shape[0]:
            scores[i] = logits[pred_pos, vocab_ids_t].cpu()
        pos += n_soft + len(tok_ids)

    return scores, vocab


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def predict(
    subject:   str,
    session:   int,
    condition: str,
    poem:      str,
    method:    str,
    ckpt_dir:  str,
    device:    str = "cpu",
) -> Dict:
    """
    Run inference for one trial and return scores over the eval vocabulary.

    Parameters
    ----------
    subject, session, condition, poem : trial identifier
    method    : 'inference' | 'twostage' | 'interleaved'
    ckpt_dir  : path to the method's output directory (contains *_best.pt files)
    device    : 'cuda' or 'cpu'

    Returns
    -------
    dict with keys:
        words      : List[str]         ground-truth word sequence
        pred_top1  : List[str]         argmax prediction per position
        scores     : Tensor(N, |V|)    raw scores over vocab
        vocab      : List[str]         sorted evaluation vocabulary
        valid      : List[bool]        True where MEG window was available
    """
    ckpt_dir = Path(ckpt_dir)
    dev      = torch.device(device)
    cfg      = _load_run_config(ckpt_dir)
    cfg["poem"] = poem                   # inject trial info for helpers

    windows, valid_mask, word_texts = _load_meg_windows(
        subject, session, condition, poem
    )

    if method == "inference":
        scores, vocab = _predict_inference(
            windows, valid_mask, word_texts, ckpt_dir, dev, cfg)
    elif method == "twostage":
        scores, vocab = _predict_twostage(
            windows, valid_mask, word_texts, ckpt_dir, dev, cfg)
    elif method == "interleaved":
        scores, vocab = _predict_interleaved(
            windows, valid_mask, word_texts, ckpt_dir, dev, cfg)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    vocab_idx  = {w: i for i, w in enumerate(vocab)}
    pred_top1  = [vocab[scores[i].argmax().item()] for i in range(len(word_texts))]

    return {
        "words":     word_texts,
        "pred_top1": pred_top1,
        "scores":    scores,          # (N, |V|)  — raw, use for fusion
        "vocab":     vocab,
        "valid":     valid_mask.tolist(),
    }
