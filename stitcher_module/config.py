"""
Author: Ktovoz
Date: 2025.11.28
"""

DEFAULT_OUTPUT_NAME: str = "stitched_result"
OUTPUT_SUFFIX: str = "_output"
MATCHES_DIR_NAME: str = "matches"
INFO_FILE_NAME: str = "stitching_info.json"

MATCH_VISUALIZATION_PATTERN: str = "match_{idx1:02d}_vs_{idx2:02d}_{name1}_vs_{name2}.jpg"


def get_match_pattern(idx1: int, idx2: int, name1: str, name2: str) -> str:
    return MATCH_VISUALIZATION_PATTERN.format(
        idx1=idx1 + 1, idx2=idx2 + 1, name1=name1, name2=name2
    )
