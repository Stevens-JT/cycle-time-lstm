import json, ast
from pyspark.sql.types import ArrayType, DoubleType

def parse_array(cell: str):
    if cell is None:
        return None
    s = cell.strip()
    if not s:
        return None
    try:
        v = json.loads(s)
    except Exception:
        try:
            v = ast.literal_eval(s)
        except Exception:
            return None
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    if isinstance(v, list):
        if len(v)==0:
            return []
        if isinstance(v[0], list):
            out = []
            for ch in v:
                out.append([to_float(x) for x in ch if x is not None])
            return out
        else:
            return [to_float(x) for x in v if x is not None]
    return None
