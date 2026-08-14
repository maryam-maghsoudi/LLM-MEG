"""
train.py — unified entry point for all three methods and evaluation schemes.

Usage examples
--------------
# Method 1, LOSO, heldout sub-01, no control
python train.py --method inference --eval_scheme loso --heldout sub-01

# Method 2, 5-fold session CV, fold 0, shuffle_time control
python train.py --method twostage --eval_scheme session_cv --fold 0 --control shuffle_time

# Method 3, heldout stimulus (last 4 lines), zero control
python train.py --method interleaved --eval_scheme stimulus --n_lines 4 --control zero

# Method 2, LOSO, reuse existing Stage 1 checkpoint
python train.py --method twostage --eval_scheme loso --heldout sub-01 --load_stage1 path/to/stage1_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))    # llm_decoder/ on path


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified MEG decoder training")

    # Required
    p.add_argument("--method",      required=True,
                   choices=["inference", "twostage", "interleaved"])
    p.add_argument("--eval_scheme", required=True,
                   choices=["loso", "session_cv", "stimulus"])

    # Split selectors
    p.add_argument("--heldout",  default=None,
                   help="Held-out subject for LOSO, e.g. sub-01")
    p.add_argument("--fold",     type=int, default=None,
                   help="Fold index 0–4 for session_cv")
    p.add_argument("--n_lines",  type=int, default=2,   choices=[2, 4],
                   help="Number of heldout lines for stimulus split")

    # Control
    p.add_argument("--control",  default="none",
                   choices=["none", "zero", "shuffle_time"])

    # Common model options
    p.add_argument("--llm_name",  default="HuggingFaceTB/SmolLM2-360M",
                   help="HuggingFace model ID (Methods 2 and 3)")
    p.add_argument("--bert_name", default="bert-base-uncased",
                   help="BERT model for Method 1")
    p.add_argument("--device",    default=None,
                   help="cuda / cpu (default: cuda if available)")
    p.add_argument("--out_root",  default=str(_HERE / "out"),
                   help="Root output directory")

    # Method 1 options
    p.add_argument("--bert_layer", type=int, default=-1,
                   help="BERT hidden layer to use as text targets (default: last)")

    # Method 2 options
    p.add_argument("--hmid_layer",   type=int, default=None,
                   help="LLM layer for Stage 1 targets (default: per-model best)")
    p.add_argument("--skip_stage2",  action="store_true")
    p.add_argument("--load_stage1",  default=None,
                   help="Path to existing stage1_best.pt; skips Stage 1 training")
    p.add_argument("--gru_hidden",   type=int, default=256)

    # Method 3 options
    p.add_argument("--meg_enc_ckpt", default=None,
                   help="Path to Method 1 meg_encoder_best.pt (required for interleaved)")
    p.add_argument("--n_soft",       type=int, default=1,
                   help="Number of soft tokens per word (Method 3)")

    # Training hyper-params (shared)
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--bs",      type=int,   default=None)
    p.add_argument("--patience",type=int,   default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
#  Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(requested):
    if requested:
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.tensor([1.0]).cuda()
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"[warn] CUDA unusable ({e}); falling back to CPU.")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
#  Split construction
# ---------------------------------------------------------------------------

def _make_splits(args):
    from unified.data.splits import (
        make_loso_splits, make_session_cv_splits, make_stimulus_splits,
    )
    if args.eval_scheme == "loso":
        if args.heldout is None:
            raise ValueError("--heldout required for loso eval_scheme")
        return make_loso_splits(args.heldout)
    if args.eval_scheme == "session_cv":
        if args.fold is None:
            raise ValueError("--fold required for session_cv eval_scheme")
        return make_session_cv_splits(args.fold)
    if args.eval_scheme == "stimulus":
        return make_stimulus_splits(args.n_lines)
    raise ValueError(f"Unknown eval_scheme: {args.eval_scheme}")


# ---------------------------------------------------------------------------
#  Output directory
# ---------------------------------------------------------------------------

def _out_dir(args) -> Path:
    root = Path(args.out_root)
    ctrl = f"_ctrl_{args.control}" if args.control != "none" else ""

    if args.eval_scheme == "loso":
        split_tag = f"loso_{args.heldout}"
    elif args.eval_scheme == "session_cv":
        split_tag = f"session_cv_fold{args.fold}"
    else:
        split_tag = f"stimulus_lines{args.n_lines}"

    if args.method == "twostage":
        model_tag = args.llm_name.replace("/", "_")
    elif args.method == "inference":
        model_tag = args.bert_name.replace("/", "_").replace("-", "_")
    else:
        model_tag = args.llm_name.replace("/", "_")

    return root / args.method / model_tag / f"{split_tag}{ctrl}"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = _resolve_device(args.device)
    splits = _make_splits(args)
    out    = _out_dir(args)
    out.mkdir(parents=True, exist_ok=True)

    # Save run config
    (out / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )

    print(f"\n{'='*60}")
    print(f"  method      : {args.method}")
    print(f"  eval_scheme : {args.eval_scheme}")
    print(f"  control     : {args.control}")
    print(f"  device      : {device}")
    print(f"  out_dir     : {out}")
    print(f"{'='*60}")

    # ── Method 1 ─────────────────────────────────────────────────────────────
    if args.method == "inference":
        from unified.methods.train_inference import train
        kwargs = dict(
            splits    = splits,
            out_dir   = out,
            device    = device,
            bert_name = args.bert_name,
            bert_layer= args.bert_layer,
            control   = args.control,
        )
        if args.lr:       kwargs["lr"]       = args.lr
        if args.epochs:   kwargs["epochs"]   = args.epochs
        if args.bs:       kwargs["batch_size"]= args.bs
        if args.patience: kwargs["patience"] = args.patience
        train(**kwargs)

    # ── Method 2 ─────────────────────────────────────────────────────────────
    elif args.method == "twostage":
        from unified.methods.train_twostage import train
        kwargs = dict(
            splits      = splits,
            out_dir     = out,
            device      = device,
            llm_name    = args.llm_name,
            hmid_layer  = args.hmid_layer,
            skip_stage2 = args.skip_stage2,
            load_stage1 = args.load_stage1,
            gru_hidden  = args.gru_hidden,
            control     = args.control,
        )
        if args.lr:       kwargs["s1_lr"] = kwargs["s2_lr"] = args.lr
        if args.epochs:   kwargs["s1_epochs"] = kwargs["s2_epochs"] = args.epochs
        if args.bs:       kwargs["s1_bs"] = args.bs
        if args.patience: kwargs["s1_patience"] = kwargs["s2_patience"] = args.patience
        train(**kwargs)

    # ── Method 3 ─────────────────────────────────────────────────────────────
    elif args.method == "interleaved":
        if args.meg_enc_ckpt is None:
            raise ValueError("--meg_enc_ckpt required for method=interleaved")
        from unified.methods.train_interleaved import train
        kwargs = dict(
            splits       = splits,
            out_dir      = out,
            device       = device,
            meg_enc_ckpt = args.meg_enc_ckpt,
            llm_name     = args.llm_name,
            n_soft       = args.n_soft,
            control      = args.control,
        )
        if args.lr:       kwargs["lr"]         = args.lr
        if args.epochs:   kwargs["epochs"]     = args.epochs
        if args.bs:       kwargs["batch_size"] = args.bs
        if args.patience: kwargs["patience"]   = args.patience
        train(**kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
