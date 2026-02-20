from typing import Tuple
import os
import faiss

from faasrag.core.args import IndexConfig
from faasrag.core.s3_utils import is_s3_uri, ensure_s3_file_local


def _unwrap_index(index: faiss.Index) -> Tuple[faiss.Index, faiss.Index]:
    """
    Returns (outer, inner) where:
      - outer is the index you should call search() on (possibly wrapped)
      - inner is the underlying index where search params live
    """
    inner = index
    while isinstance(inner, faiss.IndexPreTransform):
        inner = inner.index
    while isinstance(inner, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        inner = inner.index
    return index, inner


def _configure_search_params(inner: faiss.Index, cfg: IndexConfig) -> None:
    if isinstance(inner, faiss.IndexIVF):
        print(f"Setting IVF nprobe={cfg.nprobe}")
        inner.nprobe = int(cfg.nprobe)
    if hasattr(inner, "hnsw"):
        print(f"Setting HNSW efSearch={cfg.ef_search}")
        inner.hnsw.efSearch = int(cfg.ef_search)
    else:
        print("Index does not have HNSW layer, skipping efSearch config")


def _resolve_local_path(path: str, artifact_dir: str) -> str:
    """
    Rules:
      - s3://...     -> downloaded under artifact_dir, return local file path
      - absolute     -> use as-is
      - relative     -> interpret relative to artifact_dir
    """
    if is_s3_uri(path):
        return ensure_s3_file_local(
            path,
            local_base_dir=artifact_dir,
            mirror_key_under_base=True,
            skip_if_exists=True,
        )

    if os.path.isabs(path):
        return path

    return os.path.join(artifact_dir, path)


def load_index(cfg: IndexConfig, artifact_dir: str) -> faiss.Index:
    """
    Load a FAISS index from disk (or S3) and apply runtime search parameters.
    """
    if not cfg.path:
        raise ValueError("IndexConfig.path is empty")
    if not artifact_dir:
        raise ValueError("artifact_dir is empty (set it in config or env)")

    index_path = _resolve_local_path(cfg.path, artifact_dir)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    # 1) read from disk
    index = faiss.read_index(index_path)

    # 2) optional: move to GPU
    if cfg.use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, int(cfg.gpu_id), index)

    # 3) unwrap and configure search params
    _, inner = _unwrap_index(index)
    _configure_search_params(inner, cfg)

    return index
