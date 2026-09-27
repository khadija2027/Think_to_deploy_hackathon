"""A snapshot is immutable; only current.json is atomically replaced."""
import json
import os
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def index_directory():
    return Path(os.environ.get("RAG_INDEX_DIR", PROJECT_ROOT / "rag_index"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def current_manifest(root=None):
    path = Path(root or index_directory()) / "current.json"
    return read_json(path) if path.exists() else None


def snapshot_directory(root, manifest):
    version = manifest["version"]
    if not isinstance(version, str) or len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("Invalid index version")
    return Path(root) / "versions" / version
