import os
import json
import sqlite3
from dataclasses import asdict
from typing import Any, Dict, Optional, Iterable, List, Tuple

from faasrag.core.args import DocStoreConfig  # adjust import to your actual module
from faasrag.core.s3_utils import is_s3_uri, ensure_s3_file_local


# -------------------------
# Interfaces / classes
# -------------------------

class DocStore:
    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def get_many(self, pids: Iterable[str]) -> List[Optional[Dict[str, Any]]]:
        return [self.get(p) for p in pids]


class SqliteDocStore(DocStore):
    def __init__(self, db_path: str):
        self.db_path = db_path
        # check_same_thread=False if you will share this across threads
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT pid, title, text FROM passages WHERE pid = ?", (str(pid),))
        row = cur.fetchone()
        if row is None:
            return None
        return {"pid": row["pid"], "title": row["title"], "text": row["text"]}


class JsonlOffsetsDocStore(DocStore):
    """
    Random-access JSONL using pid->byte_offset index.
    Stores offsets in a JSON file (fine for 100k/500k).
    """
    def __init__(self, jsonl_path: str, offsets_path: str, id_key: str, title_key: str, text_key: str):
        self.jsonl_path = jsonl_path
        self.offsets_path = offsets_path
        self.id_key = id_key
        self.title_key = title_key
        self.text_key = text_key

        with open(offsets_path, "r", encoding="utf-8") as f:
            self.offsets: Dict[str, int] = json.load(f)

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        off = self.offsets.get(str(pid))
        if off is None:
            return None
        with open(self.jsonl_path, "rb") as f:
            f.seek(off)
            line = f.readline()
        if not line:
            return None
        obj = json.loads(line.decode("utf-8"))
        return {
            "pid": obj.get(self.id_key),
            "title": obj.get(self.title_key, ""),
            "text": obj.get(self.text_key, ""),
        }


class MemoryJsonlDocStore(DocStore):
    def __init__(self, jsonl_path: str, id_key: str, title_key: str, text_key: str):
        self.data: Dict[str, Dict[str, Any]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get(id_key)
                if pid is None:
                    continue
                self.data[str(pid)] = {
                    "pid": obj.get(id_key),
                    "title": obj.get(title_key, ""),
                    "text": obj.get(text_key, ""),
                }

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        return self.data.get(str(pid))


# -------------------------
# Builders
# -------------------------

def _docstore_root(artifact_dir: str, name: str, backend_kind: str) -> str:
    return os.path.join(artifact_dir, "docstores", name, backend_kind)

def _ensure_local_jsonl(source_uri: str, artifact_dir: str) -> str:
    """
    If source_uri is s3://..., download under artifact_dir (mirroring key).
    Else treat as local path (absolute or relative-to-artifact_dir).
    """
    if is_s3_uri(source_uri):
        return ensure_s3_file_local(source_uri, local_base_dir=artifact_dir, mirror_key_under_base=True, skip_if_exists=True)

    # local path
    if os.path.isabs(source_uri):
        return source_uri
    return os.path.join(artifact_dir, source_uri)

def _atomic_write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)

def build_sqlite_from_jsonl(
    jsonl_path: str,
    db_path: str,
    id_key: str,
    title_key: str,
    text_key: str,
    batch_size: int = 5000,
):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS passages (
                pid TEXT PRIMARY KEY,
                title TEXT,
                text TEXT
            )
        """)
        # PRIMARY KEY already indexes pid; explicit index not required but harmless:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_passages_pid ON passages(pid)")

        to_insert: List[Tuple[str, str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pid = obj.get(id_key)
                if pid is None:
                    continue
                title = obj.get(title_key, "") or ""
                text = obj.get(text_key, "") or ""
                to_insert.append((str(pid), title, text))

                if len(to_insert) >= batch_size:
                    conn.executemany("INSERT OR REPLACE INTO passages(pid, title, text) VALUES(?, ?, ?)", to_insert)
                    conn.commit()
                    to_insert.clear()

        if to_insert:
            conn.executemany("INSERT OR REPLACE INTO passages(pid, title, text) VALUES(?, ?, ?)", to_insert)
            conn.commit()

    finally:
        conn.close()

def build_offsets_from_jsonl(jsonl_path: str, offsets_path: str, id_key: str):
    os.makedirs(os.path.dirname(offsets_path), exist_ok=True)

    offsets: Dict[str, int] = {}
    with open(jsonl_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue
            obj = json.loads(line_str.decode("utf-8"))
            pid = obj.get(id_key)
            if pid is None:
                continue
            offsets[str(pid)] = pos

    tmp = offsets_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        json.dump(offsets, out)
    os.replace(tmp, offsets_path)


# -------------------------
# Public loader
# -------------------------

def load_docstore(docstore_cfg: DocStoreConfig, artifact_dir: str, backend: str) -> DocStore:
    """
    Ensures local availability of source JSONL and materializes/open backend docstore.
    """
    if not artifact_dir:
        raise ValueError("artifact_dir is empty")
    if not docstore_cfg.name:
        raise ValueError("docstore_cfg.name is empty")
    if not docstore_cfg.source_uri:
        raise ValueError("docstore_cfg.source_uri is empty")

    # 1) ensure source is local
    local_jsonl = _ensure_local_jsonl(docstore_cfg.source_uri, artifact_dir)
    if not os.path.exists(local_jsonl):
        raise FileNotFoundError(f"Docstore source JSONL not found: {local_jsonl}")

    id_key = docstore_cfg.source_id_key
    title_key = docstore_cfg.source_title_key
    text_key = docstore_cfg.source_text_key

    root = _docstore_root(artifact_dir, docstore_cfg.name, backend)
    os.makedirs(root, exist_ok=True)

    success = os.path.join(root, "_SUCCESS")
    # 2) backend materialize/open
    if backend == "sqlite":
        db_path = os.path.join(root, "docstore.sqlite")
        if not (os.path.exists(db_path) and os.path.exists(success)):
            print(f"[DocStore] Building sqlite docstore at {db_path} from {local_jsonl}")
            build_sqlite_from_jsonl(
                jsonl_path=local_jsonl,
                db_path=db_path,
                id_key=id_key,
                title_key=title_key,
                text_key=text_key,
            )
            _atomic_write_text(success, "ok\n")
        return SqliteDocStore(db_path)

    if backend == "jsonl_offsets":
        offsets_path = os.path.join(root, "offsets.json")
        if not (os.path.exists(offsets_path) and os.path.exists(success)):
            print(f"[DocStore] Building offsets at {offsets_path} from {local_jsonl}")
            build_offsets_from_jsonl(jsonl_path=local_jsonl, offsets_path=offsets_path, id_key=id_key)
            _atomic_write_text(success, "ok\n")
        return JsonlOffsetsDocStore(local_jsonl, offsets_path, id_key, title_key, text_key)

    if backend == "memory":
        return MemoryJsonlDocStore(local_jsonl, id_key, title_key, text_key)

    raise ValueError(f"Unknown backend_kind: {backend!r}")
