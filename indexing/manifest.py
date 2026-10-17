"""index_manifest.json: records exactly how the indexes were built; loaders refuse to load on a mismatch."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from indexing import config as default_config


class IndexIncompatible(Exception):
    """The persisted index was built with settings that differ from the current configuration."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_ids(chunk_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()


def settings_from_config(cfg: Any = default_config) -> dict:
    """The configuration values that must match the persisted index."""
    return {
        "embedding_model": cfg.MODEL_NAME, "model_revision": cfg.MODEL_REVISION, "normalized": cfg.NORMALIZE,
        "document_prefix": cfg.DOCUMENT_PREFIX, "distance": cfg.DISTANCE, "dimension": cfg.DIMENSION,
        "collection_name": cfg.COLLECTION_NAME,
    }


def write_manifest(manifest: dict, path: Path | None = None) -> Path:
    path = path or default_config.MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "built_at": manifest.get("built_at") or datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_manifest(path: Path | None = None) -> dict:
    path = path or default_config.MANIFEST_PATH
    if not path.is_file():
        raise IndexIncompatible(f"{path} is missing: no index has been built. Run `python -m indexing --rebuild`.")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_manifest(update: dict, path: Path | None = None) -> dict:
    """Write `update` over the existing manifest (used when only one index is rebuilt)."""
    path = path or default_config.MANIFEST_PATH
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update(update)
    write_manifest(current, path)
    return current


VECTOR_FIELDS = ("embedding_model", "model_revision", "normalized", "document_prefix", "distance", "dimension",
                 "collection_name")


def check_compatibility(cfg: Any = default_config, manifest: dict | None = None, *,
                        fields: tuple[str, ...] = VECTOR_FIELDS, manifest_path: Path | None = None) -> None:
    """Raise IndexIncompatible when the configured model/revision/normalization/distance differ from the
    manifest. `cfg` is any object with the config attributes (the config module by default)."""
    manifest = manifest if manifest is not None else read_manifest(manifest_path)
    want = settings_from_config(cfg)
    for key in fields:
        if key not in manifest:
            raise IndexIncompatible(f"index manifest has no '{key}'; rebuild with `python -m indexing --rebuild`")
        if manifest[key] != want[key]:
            raise IndexIncompatible(f"index was built with {key}={manifest[key]!r}, config says {want[key]!r}; "
                                    f"rebuild with `python -m indexing --rebuild`")


def config_with(**overrides) -> SimpleNamespace:
    """A copy of the current config with some attributes changed (used by tests)."""
    values = {k: getattr(default_config, k) for k in dir(default_config) if k.isupper()}
    values.update(overrides)
    return SimpleNamespace(**values)


def check_bm25_compatibility(tokenizer_version: str, cfg: Any = default_config, manifest: dict | None = None,
                             manifest_path: Path | None = None) -> None:
    """Raise IndexIncompatible when the BM25 corpus/tokenizer/settings differ from what the manifest recorded."""
    manifest = manifest if manifest is not None else read_manifest(manifest_path)
    built = manifest.get("bm25")
    if not built:
        raise IndexIncompatible("index manifest has no BM25 section; rebuild with `python -m indexing --rebuild`")
    if built.get("tokenizer_version") != tokenizer_version:
        raise IndexIncompatible(f"BM25 index was built with tokenizer_version={built.get('tokenizer_version')!r}, "
                                f"code says {tokenizer_version!r}; rebuild with `python -m indexing --rebuild`")
    for key, want in cfg.BM25_SETTINGS.items():
        if built.get(key) != want:
            raise IndexIncompatible(f"BM25 index was built with {key}={built.get(key)!r}, config says {want!r}; "
                                    f"rebuild with `python -m indexing --rebuild`")
