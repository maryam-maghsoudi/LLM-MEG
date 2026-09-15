"""
teacher_cache.py
=======================
 one-time cache of frozen teacher targets for both poems:
  - audio_target : wav2vec2 hidden states, per ~20ms frame, projected to JOINT_DIM
  - h_mid        : GPT-2 middle-layer hidden state, per word, projected to JOINT_DIM              
  - hf           : GPT-2 final hidden state, per word, projected to JOINT_DIM                 
  - hf_full      : GPT-2 final hidden state, per word, UN-projected (768d)
injection layer L instead of only the final layer:
  - hidden_states_full[L]         : (T_tokens, 768) full per-token hidden
                                     state at hidden_states-index L, UN-
                                     projected.
  - hidden_states_word_aligned[L] : (N_words, 768) the same, but subset to
                                     each word's last-subtoken position via
                                     word_to_last_subtok — the word-level
                                     "state after having seen up to word t".
Run :  python build_teacher_cache.py
Output :  teacher_cache.pt  — loaded by the training loop, never recomputed
           per epoch (both wav2vec2 and GPT-2 are frozen and deterministic).
"""

import json
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from transformers import (
        Wav2Vec2Model, Wav2Vec2FeatureExtractor,
    GPT2Model, GPT2TokenizerFast,
)

JOINT_DIM = 128
MEG_OUTPUT_RATE_HZ = 50.0   # MEG encoder's output rate 
# GPT2_SWEEP_LAYERS = [0, 4, 8, 12]
GPT2_SWEEP_LAYERS = list(range(0,13))
WAV2VEC2_SWEEP_LAYERS = list(range(0, 13))

def fixed_random_projection(dim_in: int, dim_out: int, seed: int) -> torch.Tensor:
    """
    A deterministic, UNTRAINED random projection matrix (dim_in -> dim_out).
    Orthonormalized via QR so it behaves as a near-isometry (Johnson-
    Lindenstrauss): pairwise cosine similarities in the reduced space
    approximately match the original space.

    Returns
    -------
    q  : Orthonormal matrix (dim_in, dim_out)
    """
    g = torch.Generator().manual_seed(seed)
    mat = torch.randn(dim_in, dim_out, generator=g)
    q, _ = torch.linalg.qr(mat)   # orthonormal columns
    return q   # (dim_in, dim_out)


def resample_time(x: torch.Tensor, src_hz: float, tgt_hz: float) -> torch.Tensor:
    """
    Resample a (T, D) sequence from src_hz to tgt_hz along time, via linear
    interpolation, so frame i of the resampled output lands on the same
    real-time instant as frame i of a sequence natively at tgt_hz.

    Returns
    -------
    x_resampled : (T_new, D)
    """
    T, D = x.shape
    duration_s = T / src_hz
    T_new = max(1, round(duration_s * tgt_hz))
    x_t = x.T.unsqueeze(0)                         # (1, D, T)
    x_resampled = F.interpolate(x_t, size=T_new, mode="linear", align_corners=False)
    return x_resampled.squeeze(0).T                 # (T_new, D)


def build_audio_target_from_hidden(hidden_layer: torch.Tensor, proj_audio: torch.Tensor,
                                    src_hz: float, target_rate_hz: float) -> torch.Tensor:
    resampled = resample_time(hidden_layer, src_hz=src_hz, tgt_hz=target_rate_hz)
    return resampled 


def tokenize_with_word_alignment(tokenizer, words: list[str]):
    """
    Tokenizes the full poem as one continuous string (not word-by-word —
    GPT-2's BPE merges depend on surrounding characters/spaces, so
    tokenizing words in isolation would not match how the poem is actually
    tokenized in context). Tracks which token index is each word's LAST
    subtoken, via character offsets.

    Returns
    -------
    input_ids            : LongTensor(1, T)
    word_to_last_subtok  : LongTensor(N_words,) — index into input_ids / hidden_states
    """
    text = words[0] + "".join(" " + w for w in words[1:])
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"][0].tolist()   # [(char_start, char_end), ...]

    word_char_spans = []
    cursor = 0
    for i, w in enumerate(words):
        if i > 0:
            cursor += 1  # the inserted space
        word_char_spans.append((cursor, cursor + len(w)))
        cursor += len(w)

    word_to_last_subtok = []
    for (w_start, w_end) in word_char_spans:
        last = None
        for t, (c_start, c_end) in enumerate(offsets):

            if c_end > c_start and c_end > w_start and c_start < w_end:
                last = t
        if last is None:
            raise ValueError(f"Could not align word span {(w_start, w_end)} "
                              f"('{text[w_start:w_end]}') to any token")
        word_to_last_subtok.append(last)

    return input_ids, torch.tensor(word_to_last_subtok, dtype=torch.long)


def continue_forward_from_layer(
    lm: GPT2Model,
    hidden_state: torch.Tensor,
    layer_idx: int,
    no_grad: bool = True,
) -> torch.Tensor:
    """
    Resume GPT-2's own forward computation from a hidden state that sits at
    hidden_states-index `layer_idx`.Runs the remaining blocks[layer_idx:]
    then ln_f, exactly as GPT2Model's own forward would internally.

    If layer_idx == num_layers (the final index), the input is already the
    finished, post-ln_f state — this is a no-op pass-through, and is the
    injection depth used by the current Stage 2 design (§7): no frozen
    blocks touched at all.

    hidden_state : (T, 768) or (B, T, 768) — a single trial's FULL per-token
                   sequence, not just word-aligned positions. GPT-2's causal
                   self-attention in the remaining blocks needs a real,
                   consistent prefix at every position, not an isolated
                   spliced timestep.

    no_grad : True for offline verification (this file's own usage below —
              real hidden states, nothing to train, no gradients needed).
              Set False when this function is reused inside the actual Eval
              2 training loop.

    Returns : same shape as hidden_state, GPT-2's real post-ln_f final
              hidden state (ready for `hidden_state @ lm.wte.weight.T`).
    """
    num_layers = len(lm.h)
    assert 0 <= layer_idx <= num_layers, f"layer_idx must be in [0, {num_layers}], got {layer_idx}"

    squeeze_back = hidden_state.dim() == 2
    hs = hidden_state.unsqueeze(0) if squeeze_back else hidden_state
    assert hs.dim() == 3, (
        f"continue_forward_from_layer expects hidden_state to end up 3D (batch, seq, hidden) "
        f"before entering GPT-2's blocks, got shape {tuple(hs.shape)} from input shape "
        f"{tuple(hidden_state.shape)}. This should be unreachable — if you see this, the caller "
        f"is passing something other than a plain (T,768) or (B,T,768) tensor."
    )

    if layer_idx == num_layers:
        return hidden_state  # already post-ln_f, nothing left to run

    def _run():
        out = hs
        for i, block in enumerate(lm.h[layer_idx:]):
            assert out.dim() == 3, (
                f"hidden state collapsed to shape {tuple(out.shape)} after block "
                f"{layer_idx + i - 1} (started 3D at {tuple(hs.shape)}) — this points to a "
                f"transformers-version difference in GPT2Block's forward signature, not a bug "
                f"in the shapes we're feeding it. Check transformers.__version__ and GPT2Block.forward's "
                f"signature directly."
            )
            block_out = block(hidden_states=out)
            out = block_out[0] if isinstance(block_out, tuple) else block_out
        return lm.ln_f(out) # final layer norm

    if no_grad:
        with torch.no_grad():
            result = _run()
    else:
        result = _run()

    return result.squeeze(0) if squeeze_back else result


def sanity_check_layer_continuation(lm: GPT2Model, hidden_states: tuple, sweep_layers: list[int]) -> dict:
    """
     Returns a dict keyed by layer_idx with reconstruction diagnostics.
    """
    final_idx = len(lm.h)
    true_final = hidden_states[final_idx][0]          # (T, 768), real post-ln_f
    true_logits_argmax = (true_final @ lm.wte.weight.T).argmax(dim=-1)

    results = {}
    for L in sweep_layers:
        if L == final_idx:
            results[L] = {"max_abs_error": 0.0, "argmax_agreement": 1.0, "note": "final layer, no-op"}
            continue
        recon = continue_forward_from_layer(lm, hidden_states[L][0], L, no_grad=True)
        max_abs_error = (recon - true_final).abs().max().item()
        recon_argmax = (recon @ lm.wte.weight.T).argmax(dim=-1)
        agreement = (recon_argmax == true_logits_argmax).float().mean().item()
        results[L] = {"max_abs_error": max_abs_error, "argmax_agreement": agreement}
    return results

def check_projection_reconstruction(cache_entry: dict, proj_text: torch.Tensor, layer: int = 8) -> dict:
    """
    Diagnostic (not a correctness check on existing code -- this measures
    a KNOWN, expected property of fixed_random_projection, motivated by
    wanting to eventually inject learned MEG->h_mid predictions back into
    GPT-2's real hidden-state space).

    proj_text (768, 128) has ORTHONORMAL COLUMNS (Q^T Q = I_128), but
    Q @ Q.T is NOT the 768-dim identity -- it's a rank-128 projection onto
    a RANDOM 128-dim subspace of the 768-dim space. Reconstructing
    h_hat_768 = h_mid @ Q.T therefore recovers only the component of the
    true 768-dim vector lying in that random subspace; the orthogonal
    complement (640 dims) is lost at h_mid = h_768 @ Q, before this
    function even runs -- no downstream cleverness can recover it.

    Also verifies internal consistency: h_mid should already equal
    hidden_states_word_aligned[layer] @ proj_text (both cached from the
    SAME layer, gpt2_mid_layer defaults to 8, matching sweep layer 8).
    """
    h_768 = cache_entry["hidden_states_word_aligned"][layer]   # (N_words, 768), real, unprojected
    h_mid = cache_entry["h_mid"]                                 # (N_words, 128), cached projection

    # consistency: does the cached h_mid actually match h_768 @ proj_text?
    h_mid_recomputed = h_768 @ proj_text
    consistency_max_err = (h_mid - h_mid_recomputed).abs().max().item()

    # the actual question: how much survives a round trip through Q.T?
    h_hat_768 = h_mid @ proj_text.T
    cos_sim = F.cosine_similarity(h_768, h_hat_768, dim=-1)                     # (N_words,)
    variance_retained = (h_hat_768.norm(dim=-1) ** 2) / (h_768.norm(dim=-1) ** 2)  # (N_words,)

    return {
        "layer": layer,
        "consistency_max_err": consistency_max_err,   # should be ~0.00e+00
        "cos_sim_mean": cos_sim.mean().item(),
        "cos_sim_min": cos_sim.min().item(),
        "cos_sim_max": cos_sim.max().item(),
        "variance_retained_mean": variance_retained.mean().item(),   # expect ~128/768 ≈ 0.167 for a random Q
        "variance_retained_min": variance_retained.min().item(),
        "variance_retained_max": variance_retained.max().item(),
    }


def build_poem_teacher_cache(
    audio_path: str,
    words: list[str],
    proj_audio: torch.Tensor,
    proj_text: torch.Tensor,
    target_rate_hz: float = MEG_OUTPUT_RATE_HZ,
    wav2vec2_layer: int = 6,   # starting point — sweep empirically against Stage 1 val accuracy
    gpt2_mid_layer: int = 8,   # starting point — sweep empirically; also the default Stage 1 h_mid layer
    sweep_layers: list[int] | None = None,   # defaults to GPT2_SWEEP_LAYERS
    audio_sweep_layers: list[int] | None = None
):
    if sweep_layers is None:
        sweep_layers = GPT2_SWEEP_LAYERS
    if audio_sweep_layers is None:
        audio_sweep_layers = WAV2VEC2_SWEEP_LAYERS

    # --- Audio target: wav2vec2 ---
    wav, sr = torchaudio.load(audio_path)
    print(f'Audio shape: {wav.shape}, Sampling rate: {sr}, duration: {wav.shape[1]/sr}')
    wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)   # force mono
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval()
    num_w2v_layers = w2v.config.num_hidden_layers
    assert max(audio_sweep_layers) <= num_w2v_layers, (
        f"audio_sweep_layers {audio_sweep_layers} exceeds this wav2vec2's depth "
        f"({num_w2v_layers} transformer layers) -- WAV2VEC2_SWEEP_LAYERS assumes "
        f"wav2vec2-base (12 layers); adjust if using a different checkpoint."
    )
    inputs = fe(wav.numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = w2v(**inputs, output_hidden_states=True)
    audio_hidden_states_full = {L: out.hidden_states[L].squeeze(0).clone() for L in audio_sweep_layers}
    print(f'wav2vec2 shape (native, per layer): {audio_hidden_states_full[audio_sweep_layers[0]].shape}')
#     audio_hidden = out.hidden_states[wav2vec2_layer].squeeze(0)     # (T_w2v, 768), ~50Hz native
#     print(f'wav2vec2 shape: {audio_hidden.shape}')
#     audio_hidden = resample_time(audio_hidden, src_hz=49.95, tgt_hz=target_rate_hz)
#     print(f'wav2vec2 shape: {audio_hidden.shape}')
#     audio_target = audio_hidden @ proj_audio                        # (T, JOINT_DIM)
    audio_target = build_audio_target_from_hidden(
        audio_hidden_states_full[wav2vec2_layer], proj_audio, src_hz=49.95, target_rate_hz=target_rate_hz)
    print(f'audio_target shape (resampled+projected, layer={wav2vec2_layer}): {audio_target.shape}')

    # --- Text targets: frozen GPT-2, word-aligned ---
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    lm = GPT2Model.from_pretrained("gpt2").eval()
    num_layers = len(lm.h)
    assert max(sweep_layers) <= num_layers, (
        f"sweep_layers {sweep_layers} exceeds this GPT-2's depth ({num_layers} blocks) — "
        f"GPT2_SWEEP_LAYERS assumes gpt2-small (12 blocks); adjust if using a larger checkpoint."
    )

    input_ids, word_to_last_subtok = tokenize_with_word_alignment(tok, words)
    with torch.no_grad():
        lm_out = lm(input_ids, output_hidden_states=True)

    # --- Legacy single-layer extraction (unchanged, backward-compatible) ---
    h_mid = lm_out.hidden_states[gpt2_mid_layer][0][word_to_last_subtok]   # (N_words, 768)
    hf = lm_out.hidden_states[-1][0][word_to_last_subtok]                  # (N_words, 768), post-ln_f

#     h_mid = h_mid @ proj_text     # (N_words, JOINT_DIM)
#     hf_full = hf                  # keep un-projected hf too 
#     hf = hf @ proj_text           # (N_words, JOINT_DIM)
    h_mid = h_mid      # (N_words, JOINT_DIM)
    hf_full = hf                  # keep un-projected hf too 
    hf = hf           # (N_words, JOINT_DIM)
    # ---  full per-token + word-aligned hidden states at every sweep layer ---
    # Kept UN-projected (768d, native GPT-2 space) 
    hidden_states_full = {L: lm_out.hidden_states[L][0].clone() for L in sweep_layers}          # (T_tokens, 768)
    hidden_states_word_aligned = {L: hidden_states_full[L][word_to_last_subtok].clone()
                                   for L in sweep_layers}                                        # (N_words, 768)

    # --- oracle-ceiling sanity check ---
    sanity_check = sanity_check_layer_continuation(lm, lm_out.hidden_states, sweep_layers)
    for L, diag in sanity_check.items():
        print(f"  [sanity check] layer {L}: max_abs_error={diag['max_abs_error']:.2e}, "
              f"argmax_agreement={diag.get('argmax_agreement', 1.0):.4f}")

    return {
        "audio_target": audio_target,
        "audio_hidden_states_full": audio_hidden_states_full,
        "h_mid": h_mid,
        "hf": hf,
        "hf_full": hf_full,  
        "hidden_states_full": hidden_states_full,
        "hidden_states_word_aligned": hidden_states_word_aligned,
        "input_ids": input_ids,
        "word_to_last_subtok": word_to_last_subtok,
        "sweep_layers": sweep_layers,
        "audio_sweep_layers": audio_sweep_layers,
        "sanity_check": sanity_check,
    }


def _describe(entry):
    """Handles both plain tensors and the new dict-of-tensor fields when printing final cache shapes."""
    if isinstance(entry, dict):
        return {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in entry.items()}
    if torch.is_tensor(entry):
        return tuple(entry.shape)
    return entry


if __name__ == "__main__":
    ONSET_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/contrastive_learning/onset_out"
    with open(f"{ONSET_DIR}/poem1_word_onsets.json") as f:
        poem1_json = json.load(f)
    with open(f"{ONSET_DIR}/poem2_word_onsets.json") as f:
        poem2_json = json.load(f)

    poem1_words = [item["word"] for item in poem1_json]
    poem2_words = [item["word"] for item in poem2_json]
    print(f'Poem1 no of words: {len(poem1_words)}, Poem2 no of words: {len(poem2_words)}')
    AUDIO_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/rnn/audio_wav"
    poem1_audio_path = f"{AUDIO_DIR}/poem1.wav"
    poem2_audio_path = f"{AUDIO_DIR}/poem2.wav"

    # Seeded once, saved alongside the cache — MUST be reloaded (not
    # regenerated) anywhere these projections are used again, since a
    # different seed or torch version could in principle produce a
    # different (still valid, but DIFFERENT) random basis.
    proj_audio = fixed_random_projection(768, JOINT_DIM, seed=0)
    proj_text = fixed_random_projection(768, JOINT_DIM, seed=1)

    print("poem1:")
    poem1_cache = build_poem_teacher_cache(poem1_audio_path, poem1_words, proj_audio, proj_text)
    print("poem2:")
    poem2_cache = build_poem_teacher_cache(poem2_audio_path, poem2_words, proj_audio, proj_text)

    cache = {
        "poem1": poem1_cache,
        "poem2": poem2_cache,
        "proj_audio": proj_audio,
        "proj_text": proj_text,
        "sweep_layers": GPT2_SWEEP_LAYERS,
    }
    torch.save(cache, "teacher_cache.pt")
    print({p: {k: _describe(v) for k, v in d.items()} for p, d in cache.items() if p in ("poem1", "poem2")})

#      # --projection reconstruction diagnostic ---
#     print("\n=== projection reconstruction check (layer 8, matches h_mid's gpt2_mid_layer default) ===")
#     for poem_name, poem_cache in (("poem1", poem1_cache), ("poem2", poem2_cache)):
#         diag = check_projection_reconstruction(poem_cache, proj_text, layer=8)
#         print(f"  {poem_name}: consistency_err={diag['consistency_max_err']:.2e}  "
#               f"cos_sim mean={diag['cos_sim_mean']:.3f} (min={diag['cos_sim_min']:.3f}, max={diag['cos_sim_max']:.3f})  "
#               f"variance_retained mean={diag['variance_retained_mean']:.3f} "
#               f"(min={diag['variance_retained_min']:.3f}, max={diag['variance_retained_max']:.3f})")
#     lm = GPT2Model.from_pretrained("gpt2").eval()
#     dummy = torch.randn(5, 768)   # fake (T, hidden) — no MEG or GPT-2 forward needed
#     out = continue_forward_from_layer(lm, dummy, layer_idx=8, no_grad=True)
#     print(out.shape)   # should print torch.Size([5, 768])
