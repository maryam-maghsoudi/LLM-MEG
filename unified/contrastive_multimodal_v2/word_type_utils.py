"""
Shared word-type utilities. Single source of truth for "which word type
does this occurrence-label correspond to" and "how frequent is that type."


"""
from collections import Counter
from typing import Dict, List, Tuple


def _load_word_sequence(onset_json_path: str) -> List[str]:
    import json
    with open(onset_json_path) as f:
        entries = json.load(f)
    return [e["word"] for e in entries]  


def build_word_type_maps(
    onset_json_paths: Dict[int, str],   # {poem_id: path}
    max_word_pos: int,                  # MUST match llm_contrastive_loss's label formula exactly
) -> Tuple[Dict[int, str], Counter]:
    """
    label_to_type: {poem_id*max_word_pos + word_pos -> word_type_string}
    type_freq:     word_type_string -> total occurrences across both poems
    """
    label_to_type: Dict[int, str] = {}
    type_freq: Counter = Counter()
    for poem_id, path in onset_json_paths.items():
        for word_pos, word in enumerate(_load_word_sequence(path)):
            label = poem_id * max_word_pos + word_pos
            label_to_type[label] = word
            type_freq[word] += 1
    return label_to_type, type_freq
