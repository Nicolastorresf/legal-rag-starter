import re
import yaml
from typing import List, Dict

def load_config(path="src/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_txt(x: str) -> str:
    if x is None: return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def expand_query(q: str, synonyms: Dict[str, List[str]]) -> List[str]:
    qn = q.lower().strip()
    expanded = {qn}
    for key, syns in (synonyms or {}).items():
        key_l = key.lower()
        if key_l in qn:
            expanded.update([s.lower() for s in syns])
    return list(expanded)
