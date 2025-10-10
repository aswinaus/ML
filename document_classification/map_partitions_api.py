from typing import Iterator, Dict, Any, List
import uuid

def map_partitions_api(rows_iter: Iterator[Dict[str,Any]], fn):
    # rows_iter is an iterator of dicts; fn returns dict/list
    for row in rows_iter:
        try:
            out = fn(row)
            if isinstance(out, list):
                for o in out: yield o
            else:
                yield out
        except Exception as e:
            yield {"_error": str(e), "_row": row, "_id": str(uuid.uuid4())}
