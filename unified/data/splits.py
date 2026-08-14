"""
splits.py — train/val/test split functions for three evaluation schemes.

All functions return a dict with keys 'train', 'val', 'test', each being:
    {
        'trials':      [(subject, poem, session), ...],
        'word_filter': {poem: [word_positions]} or None   # None = all positions
    }

A trial is the atomic split unit — all word windows from one
(subject, poem, session) go exclusively to one partition. See DESIGN.md §4.
"""

from itertools import product
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

SUBJECTS: List[str] = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
POEM_KEYS:  List[str] = ["poem1", "poem2"]
N_SESSIONS: int       = 10

# Word positions per line (1-indexed lines), derived from WhisperX forced
# alignment using a 0.8 s gap threshold. See DESIGN.md §2.2.
POEM_LINES: Dict[str, List[List[int]]] = {
    "poem1": [
        list(range(0,  5)),   # L1:  when out on the lawn
        list(range(5,  10)),  # L2:  there arose such a clatter
        list(range(10, 15)),  # L3:  i sprang from my bed
        list(range(15, 21)),  # L4:  to see what was the matter
        list(range(21, 25)),  # L5:  away to the window
        list(range(25, 30)),  # L6:  i flew like a flash
        list(range(30, 34)),  # L7:  tore open the shutters
        list(range(34, 39)),  # L8:  and threw up the sash
        list(range(39, 44)),  # L9:  the moon on the breast
        list(range(44, 48)),  # L10: of the new-fallen snow
        list(range(48, 53)),  # L11: gave a lustre of midday
        list(range(53, 56)),  # L12: to objects below
    ],
    "poem2": [
        list(range(0,  6)),   # L1:  he was dressed all in fur
        list(range(6,  12)),  # L2:  from his head to his foot
        list(range(12, 18)),  # L3:  and his clothes were all tarnished
        list(range(18, 22)),  # L4:  with ashes and soot
        list(range(22, 26)),  # L5:  a bundle of toys
        list(range(26, 32)),  # L6:  he had flung on his back
        list(range(32, 38)),  # L7:  and he looked like a peddler
        list(range(38, 42)),  # L8:  just opening his pack
        list(range(42, 47)),  # L9:  his eyes how they twinkled
        list(range(47, 51)),  # L10: his dimples how merry
        list(range(51, 56)),  # L11: his cheeks were like roses
        list(range(56, 61)),  # L12: his nose like a cherry
    ],
}

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

Trial = Tuple[str, str, int]   # (subject, poem, session)


def _trials(subjects: List[str], sessions: List[int]) -> List[Trial]:
    return [(s, p, sess) for s, p, sess in product(subjects, POEM_KEYS, sessions)]


def _fmt(split: Dict) -> str:
    n = len(split["trials"])
    wf = "all words" if split["word_filter"] is None else "filtered words"
    return f"{n} trials, {wf}"


# ---------------------------------------------------------------------------
#  Split functions
# ---------------------------------------------------------------------------

def make_loso_splits(
    heldout_subject: str,
    all_subjects:    List[str] = SUBJECTS,
    train_sessions:  List[int] = list(range(8)),
    val_sessions:    List[int] = [8, 9],
) -> Dict:
    """
    Leave-one-subject-out.
      train : 12 non-heldout subjects × both poems × sessions 0–7
      val   : 12 non-heldout subjects × both poems × sessions 8–9
      test  : heldout subject          × both poems × sessions 0–9
    """
    if heldout_subject not in all_subjects:
        raise ValueError(
            f"{heldout_subject!r} not in SUBJECTS list.\nValid: {all_subjects}"
        )
    train_subs = [s for s in all_subjects if s != heldout_subject]
    splits = {
        "train": {"trials": _trials(train_subs,        train_sessions),         "word_filter": None},
        "val":   {"trials": _trials(train_subs,        val_sessions),           "word_filter": None},
        "test":  {"trials": _trials([heldout_subject], list(range(N_SESSIONS))), "word_filter": None},
    }
    print(f"LOSO split  heldout={heldout_subject}")
    for k, v in splits.items():
        print(f"  {k:5s}: {_fmt(v)}")
    return splits


def make_session_cv_splits(
    fold_k:       int,
    all_subjects: List[str] = SUBJECTS,
    n_folds:      int = 5,
) -> Dict:
    """
    5-fold cross-validation over sessions.
    Sessions are grouped in consecutive pairs; fold k tests [2k, 2k+1].
    Val = previous fold's sessions (cyclic). Train = remaining 6 sessions.

    fold | test   | val    | train
    -----|--------|--------|--------------------
      0  | [0,1]  | [8,9]  | [2,3,4,5,6,7]
      1  | [2,3]  | [0,1]  | [4,5,6,7,8,9]
      2  | [4,5]  | [2,3]  | [0,1,6,7,8,9]
      3  | [6,7]  | [4,5]  | [0,1,2,3,8,9]
      4  | [8,9]  | [6,7]  | [0,1,2,3,4,5]
    """
    if not (0 <= fold_k < n_folds):
        raise ValueError(f"fold_k must be 0–{n_folds - 1}, got {fold_k}")

    test_sessions  = [2 * fold_k,              2 * fold_k + 1]
    prev           = (fold_k - 1) % n_folds
    val_sessions   = [2 * prev,                2 * prev + 1]
    train_sessions = [s for s in range(N_SESSIONS)
                      if s not in test_sessions and s not in val_sessions]

    splits = {
        "train": {"trials": _trials(all_subjects, train_sessions), "word_filter": None},
        "val":   {"trials": _trials(all_subjects, val_sessions),   "word_filter": None},
        "test":  {"trials": _trials(all_subjects, test_sessions),  "word_filter": None},
    }
    print(f"Session CV  fold={fold_k}  test={test_sessions}  val={val_sessions}  "
          f"train={train_sessions}")
    for k, v in splits.items():
        print(f"  {k:5s}: {_fmt(v)}")
    return splits


def make_stimulus_splits(
    n_heldout_lines: int = 2,
    all_subjects:    List[str] = SUBJECTS,
    train_sessions:  List[int] = list(range(8)),
    val_sessions:    List[int] = [8, 9],
) -> Dict:
    """
    Hold out the last n_heldout_lines lines of each poem by word position.
    Trial split is session-based; word_filter restricts which positions are included.

    n_heldout_lines=2 → test = lines 11–12
    n_heldout_lines=4 → test = lines  9–12

    train word filter : lines 1..(12-n), sessions 0–7
    val   word filter : lines 1..(12-n), sessions 8–9   [same lines, different sessions]
    test  word filter : last n lines,    all sessions
    """
    if n_heldout_lines not in (2, 4):
        raise ValueError("n_heldout_lines must be 2 or 4")

    n_train_lines = 12 - n_heldout_lines
    all_sessions  = list(range(N_SESSIONS))

    train_filter: Dict[str, List[int]] = {}
    test_filter:  Dict[str, List[int]] = {}
    for poem, lines in POEM_LINES.items():
        train_filter[poem] = [w for ln in lines[:n_train_lines] for w in ln]
        test_filter[poem]  = [w for ln in lines[n_train_lines:]  for w in ln]

    splits = {
        "train": {"trials": _trials(all_subjects, train_sessions), "word_filter": train_filter},
        "val":   {"trials": _trials(all_subjects, val_sessions),   "word_filter": train_filter},
        "test":  {"trials": _trials(all_subjects, all_sessions),   "word_filter": test_filter},
    }
    test_lines = list(range(n_train_lines + 1, 13))
    print(f"Stimulus split  n_heldout_lines={n_heldout_lines}  test_lines={test_lines}")
    for k, v in splits.items():
        print(f"  {k:5s}: {_fmt(v)}")
    return splits
