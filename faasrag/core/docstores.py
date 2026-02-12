import os
import json
import sqlite3
from typing import Any, Dict, Optional, Iterable, List, Tuple, Protocol

import boto3
from botocore.config import Config

from faasrag.core.args import DocStoreConfig  # adjust import to your actual module
from faasrag.core.s3_utils import is_s3_uri, ensure_s3_file_local


# -------------------------
# Interfaces
# -------------------------

class DocStore:
    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def get_many(self, pids: Iterable[str]) -> List[Optional[Dict[str, Any]]]:
        return [self.get(p) for p in pids]


class ByteOffsetLineReader(Protocol):
    """Return the JSONL line (bytes) that begins at byte offset `off`, without trailing newline."""
    def read_line_at(self, off: int) -> Optional[bytes]:
        ...


# -------------------------
# Readers (local vs S3)
# -------------------------

class LocalByteReader(ByteOffsetLineReader):
    def __init__(self, path: str):
        self.path = path
        self._f = open(self.path, "rb")

    def close(self) -> None:
        if getattr(self, "_f", None) is not None:
            try:
                self._f.close()
            finally:
                self._f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read_line_at(self, off: int) -> Optional[bytes]:
        self._f.seek(off)
        line = self._f.readline()
        if not line:
            return None
        return line.strip()  # strip newline


class S3RangeByteReader(ByteOffsetLineReader):
    """
    Reads a JSONL line from S3 using Range GET. No caching.
    """
    def __init__(
        self,
        s3_uri: str,
        *,
        chunk_bytes: int = 64 * 1024,
        max_chunk_bytes: int = 1024 * 1024,
        boto3_config: Optional[Config] = None,
        max_pool_connections: int = 32,
    ):
        if not is_s3_uri(s3_uri):
            raise ValueError(f"Expected s3://... URI, got {s3_uri!r}")

        self.bucket, self.key = self._parse_s3_uri(s3_uri)
        self.chunk_bytes = int(chunk_bytes)
        self.max_chunk_bytes = int(max_chunk_bytes)

        if self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")
        if self.max_chunk_bytes < self.chunk_bytes:
            raise ValueError("max_chunk_bytes must be >= chunk_bytes")

        cfg = boto3_config or Config(
            retries={"max_attempts": 10, "mode": "standard"},
            max_pool_connections=max_pool_connections,
        )
        self.s3 = boto3.client("s3", config=cfg)

    @staticmethod
    def _parse_s3_uri(uri: str) -> Tuple[str, str]:
        no_scheme = uri[len("s3://") :]
        parts = no_scheme.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid s3 uri: {uri!r}")
        return parts[0], parts[1]

    def _range_get(self, start: int, length: int) -> bytes:
        end = start + length - 1
        resp = self.s3.get_object(
            Bucket=self.bucket,
            Key=self.key,
            Range=f"bytes={start}-{end}",
        )
        return resp["Body"].read()

    def read_line_at(self, off: int) -> Optional[bytes]:
        length = self.chunk_bytes
        while True:
            data = self._range_get(off, length)
            if not data:
                return None

            nl = data.find(b"\n")
            if nl != -1:
                line = data[:nl].strip()
                return line if line else None

            if length >= self.max_chunk_bytes:
                # last-ditch: treat entire buffer as a line
                line = data.strip()
                return line if line else None

            length = min(self.max_chunk_bytes, length * 2)


# -------------------------
# DocStores
# -------------------------

class LocalSqliteDocStore(DocStore):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT pid, title, text FROM passages WHERE pid = ?", (str(pid),))
        row = cur.fetchone()
        if row is None:
            return None
        return {"pid": row["pid"], "title": row["title"], "text": row["text"]}


class OffsetsJsonlDocStore(DocStore):
    """
    Generic offsets docstore. Works with local JSONL or S3 JSONL depending on the reader.
    """
    def __init__(
        self,
        *,
        offsets_path: str,
        reader: ByteOffsetLineReader,
        id_key: str,
        title_key: str,
        text_key: str,
        sort_get_many_by_offset: bool = True,
    ):
        self.reader = reader
        self.id_key = id_key
        self.title_key = title_key
        self.text_key = text_key
        self.sort_get_many_by_offset = sort_get_many_by_offset

        with open(offsets_path, "r", encoding="utf-8") as f:
            self.offsets: Dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}

    def _decode(self, line: bytes) -> Optional[Dict[str, Any]]:
        if not line:
            return None
        obj = json.loads(line.decode("utf-8"))
        return {
            "pid": obj.get(self.id_key),
            "title": obj.get(self.title_key, "") or "",
            "text": obj.get(self.text_key, "") or "",
        }

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        off = self.offsets.get(str(pid))
        if off is None:
            return None
        line = self.reader.read_line_at(off)
        return self._decode(line) if line else None

    def get_many(self, pids: Iterable[str]) -> List[Optional[Dict[str, Any]]]:
        pid_list = [str(p) for p in pids]
        pairs: List[Tuple[str, int]] = [(p, self.offsets[p]) for p in pid_list if p in self.offsets]

        if self.sort_get_many_by_offset:
            pairs.sort(key=lambda x: x[1])

        results: Dict[str, Optional[Dict[str, Any]]] = {p: None for p in pid_list}
        for p, off in pairs:
            line = self.reader.read_line_at(off)
            results[p] = self._decode(line) if line else None
        return [results[p] for p in pid_list]


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
                    "title": obj.get(title_key, "") or "",
                    "text": obj.get(text_key, "") or "",
                }

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        return self.data.get(str(pid))


# -------------------------
# Builders / utils
# -------------------------

def _docstore_root(artifact_dir: str, name: str, backend: str) -> str:
    # backend name is now part of artifact layout, so keep it stable and explicit
    return os.path.join(artifact_dir, "docstores", name, backend)

def _ensure_local_jsonl(source_uri: str, artifact_dir: str) -> str:
    """
    If source_uri is s3://..., download under artifact_dir (mirroring key).
    Else treat as local path (absolute or relative-to-artifact_dir).
    """
    if is_s3_uri(source_uri):
        return ensure_s3_file_local(
            source_uri,
            local_base_dir=artifact_dir,
            mirror_key_under_base=True,
            skip_if_exists=True,
        )
    if os.path.isabs(source_uri):
        return source_uri
    return os.path.join(artifact_dir, source_uri)

def _atomic_write_text(path: str, content: str) -> None:
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
) -> None:
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

def build_offsets_from_jsonl(jsonl_path: str, offsets_path: str, id_key: str) -> None:
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
        json.dump(offsets, out, separators=(",", ":"))
    os.replace(tmp, offsets_path)


# -------------------------
# Public loader (renamed backends + offsets in artifacts)
# -------------------------

def load_docstore(docstore_cfg: DocStoreConfig, artifact_dir: str, backend: str) -> DocStore:
    """
    Backends (recommended names):
      - local_sqlite
      - local_jsonl_offsets
      - s3_jsonl_offsets
      - memory_jsonl
    """
    if not artifact_dir:
        raise ValueError("artifact_dir is empty")
    if not docstore_cfg.name:
        raise ValueError("docstore_cfg.name is empty")
    if not docstore_cfg.source_uri:
        raise ValueError("docstore_cfg.source_uri is empty")

    id_key = docstore_cfg.source_id_key
    title_key = docstore_cfg.source_title_key
    text_key = docstore_cfg.source_text_key

    root = _docstore_root(artifact_dir, docstore_cfg.name, backend)
    os.makedirs(root, exist_ok=True)
    success = os.path.join(root, "_SUCCESS")

    # ---- local_sqlite ----
    if backend == "local_sqlite":
        local_jsonl = _ensure_local_jsonl(docstore_cfg.source_uri, artifact_dir)
        if not os.path.exists(local_jsonl):
            raise FileNotFoundError(f"Docstore source JSONL not found: {local_jsonl}")

        db_path = os.path.join(root, "docstore.sqlite")
        if not (os.path.exists(db_path) and os.path.exists(success)):
            print(f"[DocStore] Building sqlite docstore at {db_path} from {local_jsonl}")
            build_sqlite_from_jsonl(local_jsonl, db_path, id_key, title_key, text_key)
            _atomic_write_text(success, "ok\n")
        return LocalSqliteDocStore(db_path)

    # ---- local_jsonl_offsets ----
    if backend == "local_jsonl_offsets":
        local_jsonl = _ensure_local_jsonl(docstore_cfg.source_uri, artifact_dir)
        if not os.path.exists(local_jsonl):
            raise FileNotFoundError(f"Docstore source JSONL not found: {local_jsonl}")

        offsets_path = os.path.join(root, "offsets.json")  # stored in artifacts
        if not (os.path.exists(offsets_path) and os.path.exists(success)):
            print(f"[DocStore] Building offsets at {offsets_path} from {local_jsonl}")
            build_offsets_from_jsonl(local_jsonl, offsets_path, id_key)
            _atomic_write_text(success, "ok\n")

        return OffsetsJsonlDocStore(
            offsets_path=offsets_path,
            reader=LocalByteReader(local_jsonl),
            id_key=id_key,
            title_key=title_key,
            text_key=text_key,
            sort_get_many_by_offset=True,
        )

    # ---- s3_jsonl_offsets ----
    if backend == "s3_jsonl_offsets":
        if not is_s3_uri(docstore_cfg.source_uri):
            raise ValueError(f"s3_jsonl_offsets requires s3:// source_uri, got {docstore_cfg.source_uri!r}")

        offsets_path = os.path.join(root, "offsets.json")  # stored in artifacts
        if not (os.path.exists(offsets_path) and os.path.exists(success)):
            # Build offsets once by downloading JSONL (or precompute offline and place offsets.json here).
            local_jsonl = _ensure_local_jsonl(docstore_cfg.source_uri, artifact_dir)
            if not os.path.exists(local_jsonl):
                raise FileNotFoundError(f"Docstore source JSONL not found: {local_jsonl}")
            print(f"[DocStore] Building offsets at {offsets_path} from {local_jsonl}")
            build_offsets_from_jsonl(local_jsonl, offsets_path, id_key)
            _atomic_write_text(success, "ok\n")

        return OffsetsJsonlDocStore(
            offsets_path=offsets_path,
            reader=S3RangeByteReader(docstore_cfg.source_uri),
            id_key=id_key,
            title_key=title_key,
            text_key=text_key,
            # Sorting by offset is harmless; you can set False if you prefer.
            sort_get_many_by_offset=True,
        )

    # ---- memory_jsonl ----
    if backend == "memory_jsonl":
        local_jsonl = _ensure_local_jsonl(docstore_cfg.source_uri, artifact_dir)
        if not os.path.exists(local_jsonl):
            raise FileNotFoundError(f"Docstore source JSONL not found: {local_jsonl}")
        return MemoryJsonlDocStore(local_jsonl, id_key, title_key, text_key)

    raise ValueError(f"Unknown backend: {backend!r}")
