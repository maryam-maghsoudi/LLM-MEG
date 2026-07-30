"""
config.py
=========
All hyperparameters and paths for the LLM-guided MEG decoder.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
MEG_BASE  = Path("/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed")
ONSET_DIR = REPO_ROOT.parent / "contrastive_learning" / "onset_out"
MEG_CKPT  = (
    REPO_ROOT.parent
    / "contrastive_learning"
    / "compare_out"
    / "models"
    / "bert_wav2vec"
    / "meg_encoder.pt"
)
OUT_DIR = REPO_ROOT / "out"

# ── Subjects & stimuli ──────────────────────────────────────────────────────
SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
POEM_KEYS  = ["poem1", "poem2"]
N_SESSIONS = 10

# ── MEG preprocessing ──────────────────────────────────────────────────────
DS_FACTOR    = 10
SFREQ_DS     = 100.0      # Hz after downsampling
N_CHANNELS   = 155
EPOCH_TMIN_S = 0.0        # epoch t=0 is stimulus onset

WIN_PRE_MS  = 200
WIN_POST_MS = 800
WIN_PRE     = int(WIN_PRE_MS  * SFREQ_DS / 1000)   # 20 samples
WIN_POST    = int(WIN_POST_MS * SFREQ_DS / 1000)   # 80 samples
WIN_SIZE    = WIN_PRE + WIN_POST                    # 100 samples

# ── Existing MEG encoder (contrastive-trained, used frozen in Option A) ────
MEG_EMB_DIM  = 128
MEG_ENC_SIZE = "small"    # matches the bert_wav2vec checkpoint

# ── LLM ────────────────────────────────────────────────────────────────────
# "gpt2"              — 117M params, d_model=768,  fastest, weakest language prior
# "gpt2-medium"       — 345M params, d_model=1024, good balance
# "microsoft/phi-2"   — 2.7B params, d_model=2560, strongest but needs more VRAM

LLM_NAME    = "gpt2"      # or "microsoft/phi-2" for more capacity
LLM_D_MODEL = 768         # GPT-2=768; if switching to Phi-3-mini set to 2048

# ── Adapter (MEG embedding → LLM token space) ──────────────────────────────
ADAPTER_HIDDEN = 512
N_SOFT_TOKENS  = 1        # soft tokens injected per MEG word window

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE   = 4          # full poem trials per batch
LR           = 1e-4
WARMUP_STEPS = 50
N_EPOCHS     = 50
PATIENCE     = 8
WEIGHT_DECAY = 1e-4
SEED         = 42

# ── Sequence design ─────────────────────────────────────────────────────────
SEQUENCE_DESIGN = "interleaved"   # "interleaved" (Design A) or "upfront" (Design B)

# ── Data splits (poem-level to guard against LLM memorisation) ─────────────
TRAIN_POEMS    = ["poem1"]
TEST_POEMS     = ["poem2"]
VAL_SESSIONS   = [8, 9]          # held-out sessions within training subjects
TRAIN_SESSIONS = list(range(8))  # sessions 0-7 for training
