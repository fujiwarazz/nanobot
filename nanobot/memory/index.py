"""Memory index manager for hybrid search over memory and sessions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.memory.chunking import chunk_text
from nanobot.memory.embeddings import EmbeddingClient


@dataclass
class MemoryFileEntry:
    path: str
    abs_path: Path
    mtime: float
    size: int
    hash: str
    content: str
    source: str


class MemoryIndexManager:
    def __init__(
        self,
        workspace_dir: Path,
        config: "MemoryConfig",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        from nanobot.config.schema import MemoryConfig
        self.workspace_dir = workspace_dir
        self.config: MemoryConfig = config
        self.db_path = Path(self.config.store.path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._watch_thread: threading.Thread | None = None

        provider = self.config.provider
        self._embedder = EmbeddingClient(
            api_key=provider.api_key or api_key,
            api_base=provider.api_base or api_base,
            model=provider.model,
            dimensions=provider.dimensions,
            encoding_format=provider.encoding_format,
            batch_size=provider.batch_size,
            max_input_chars=provider.max_input_chars,
        )

        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    path TEXT,
                    chunk_index INTEGER,
                    start_line INTEGER,
                    end_line INTEGER,
                    text TEXT,
                    hash TEXT UNIQUE,
                    embedding TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    source TEXT,
                    hash TEXT,
                    mtime REAL,
                    size INTEGER
                )
                """
            )
            if self.config.store.fts:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text, content='chunks', content_rowid='id');
                    """
                )
            conn.commit()

    def start(self) -> None:
        if self.config.sync.watch and self._watch_thread is None:
            self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._watch_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2)

    def sync(self, reason: str = "manual") -> None:
        logger.info(f"Memory sync: {reason}")
        needs_reindex = True
        meta = self._read_meta()
        if meta:
            needs_reindex = False

        if "memory" in self.config.sources:
            self._sync_memory_files(needs_reindex)
        if "sessions" in self.config.sources:
            self._sync_session_files(needs_reindex)
        if self.config.extra_paths:
            self._sync_extra_paths(needs_reindex)

        if needs_reindex:
            self._write_meta()

    def search(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        if self.config.sync.on_search:
            self.sync(reason="search")

        if not query.strip():
            return []

        limit = limit or self.config.query.max_results
        results: list[dict[str, Any]] = []

        text_candidates = self._text_search(query, limit * int(self.config.query.hybrid.candidate_multiplier))

        if not self._embedder.is_configured() or not self.config.query.hybrid.enabled:
            return text_candidates[:limit]

        # Compute embeddings for query and re-rank
        query_vecs = self._embedder.embed_texts([query])
        if not query_vecs:
            return text_candidates[:limit]
        qv = query_vecs[0]

        scored: list[tuple[float, dict[str, Any]]] = []
        for cand in text_candidates:
            emb = cand.get("embedding")
            if emb is None:
                score = cand.get("score", 0.0)
            else:
                score = (
                    self.config.query.hybrid.vector_weight * _cosine(qv, emb)
                    + self.config.query.hybrid.text_weight * cand.get("score", 0.0)
                )
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored[:limit]]
        return results

    def get_chunk(self, chunk_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, source, path, start_line, end_line, text, embedding FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "source": row[1],
            "path": row[2],
            "start_line": row[3],
            "end_line": row[4],
            "text": row[5],
            "embedding": json.loads(row[6]) if row[6] else None,
        }

    def _text_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            if self.config.store.fts:
                rows = conn.execute(
                    """
                    SELECT chunks.id, chunks.source, chunks.path, chunks.start_line, chunks.end_line,
                           chunks.text, chunks.embedding, bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks ON chunks_fts.rowid = chunks.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY score ASC
                    LIMIT ?;
                    """,
                    (query, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, source, path, start_line, end_line, text, embedding, 0.0 FROM chunks LIMIT ?",
                    (limit,),
                ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "source": row[1],
                    "path": row[2],
                    "start_line": row[3],
                    "end_line": row[4],
                    "text": row[5],
                    "embedding": json.loads(row[6]) if row[6] else None,
                    "score": 1.0 / (1.0 + row[7]) if row[7] is not None else 0.0,
                }
            )
        return results

    def _sync_memory_files(self, needs_reindex: bool) -> None:
        memory_dir = self.workspace_dir / "memory"
        if not memory_dir.exists():
            return
        files = list(memory_dir.glob("*.md"))
        self._sync_files(files, source="memory", needs_reindex=needs_reindex)

    def _sync_session_files(self, needs_reindex: bool) -> None:
        sessions_dir = Path.home() / ".nanobot" / "sessions"
        if not sessions_dir.exists():
            return
        files = list(sessions_dir.glob("*.jsonl"))
        self._sync_files(files, source="sessions", needs_reindex=needs_reindex)

    def _sync_extra_paths(self, needs_reindex: bool) -> None:
        paths: list[Path] = []
        for p in self.config.extra_paths:
            path = Path(p).expanduser()
            if path.is_dir():
                paths.extend(list(path.glob("**/*")))
            elif path.is_file():
                paths.append(path)
        self._sync_files(paths, source="extra", needs_reindex=needs_reindex)

    def _sync_files(self, files: list[Path], source: str, needs_reindex: bool) -> None:
        entries: list[MemoryFileEntry] = []
        for path in files:
            if not path.is_file():
                continue
            if path.suffix not in (".md", ".txt", ".jsonl") and source == "extra":
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            digest = _sha256(content)
            entries.append(
                MemoryFileEntry(
                    path=str(path.relative_to(self.workspace_dir)) if path.is_relative_to(self.workspace_dir) else str(path),
                    abs_path=path,
                    mtime=path.stat().st_mtime,
                    size=path.stat().st_size,
                    hash=digest,
                    content=_normalize_session_content(content, source, path),
                    source=source,
                )
            )

        active = {e.path for e in entries}
        for entry in entries:
            if not needs_reindex and self._file_hash_matches(entry):
                continue
            self._index_file(entry)
        self._delete_stale(source, active)

    def _index_file(self, entry: MemoryFileEntry) -> None:
        chunks = chunk_text(
            entry.content,
            target_tokens=self.config.chunking.tokens,
            overlap_tokens=self.config.chunking.overlap,
        )
        if not chunks:
            return
        hashes = [
            _sha256(f"{entry.hash}:{c['start_line']}:{c['end_line']}:{c['text']}")
            for c in chunks
        ]
        embeddings = self._resolve_embeddings(hashes, [c["text"] for c in chunks])

        with self._lock, sqlite3.connect(self.db_path) as conn:
            for i, chunk in enumerate(chunks):
                emb = embeddings[i] if i < len(embeddings) else None
                emb_text = json.dumps(emb) if emb is not None else None
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (source, path, chunk_index, start_line, end_line, text, hash, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.source,
                        entry.path,
                        i,
                        chunk["start_line"],
                        chunk["end_line"],
                        chunk["text"],
                        hashes[i],
                        emb_text,
                    ),
                )
            conn.execute(
                "INSERT OR REPLACE INTO files (path, source, hash, mtime, size) VALUES (?, ?, ?, ?, ?)",
                (entry.path, entry.source, entry.hash, entry.mtime, entry.size),
            )
            if self.config.store.fts:
                conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")
            conn.commit()

    def _resolve_embeddings(self, hashes: list[str], texts: list[str]) -> list[list[float]]:
        if not self._embedder.is_configured():
            return []

        cached: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        missing_hashes: list[str] = []

        with sqlite3.connect(self.db_path) as conn:
            for h in hashes:
                row = conn.execute("SELECT embedding FROM chunks WHERE hash = ?", (h,)).fetchone()
                if row and row[0]:
                    cached[h] = json.loads(row[0])
                else:
                    missing_hashes.append(h)

        for h, t in zip(hashes, texts):
            if h in cached:
                continue
            missing_texts.append(t)

        if missing_texts:
            embeddings = self._embedder.embed_texts(missing_texts)
            for h, e in zip(missing_hashes, embeddings):
                cached[h] = e

        return [cached.get(h, []) for h in hashes]

    def _file_hash_matches(self, entry: MemoryFileEntry) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT hash, mtime, size FROM files WHERE path = ?",
                (entry.path,),
            ).fetchone()
        if not row:
            return False
        return row[0] == entry.hash and row[1] == entry.mtime and row[2] == entry.size

    def _delete_stale(self, source: str, active_paths: set[str]) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, path FROM chunks WHERE source = ?",
                (source,),
            ).fetchall()
            stale_ids = [row[0] for row in rows if row[1] not in active_paths]
            if stale_ids:
                conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in stale_ids])
                conn.executemany("DELETE FROM files WHERE path = ?", [(row[1],) for row in rows if row[1] not in active_paths])
                if self.config.store.fts:
                    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")
                conn.commit()

    def _read_meta(self) -> dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return {k: v for k, v in rows}

    def _write_meta(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM meta")
            conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("indexed_at", str(time.time())))
            conn.commit()

    def _watch_loop(self) -> None:
        last_run = 0.0
        while not self._stop_event.is_set():
            now = time.time()
            if now - last_run >= self.config.sync.watch_interval_seconds:
                last_run = now
                try:
                    self.sync(reason="watch")
                except Exception as exc:
                    logger.warning(f"Memory watch sync failed: {exc}")
            time.sleep(self.config.sync.watch_debounce_seconds)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_session_content(content: str, source: str, path: Path) -> str:
    if source != "sessions":
        return content
    # For sessions, strip JSONL metadata and keep role/content lines
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("_type") == "metadata":
            continue
        role = obj.get("role")
        text = obj.get("content")
        if role and text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    denom_a = sum(x * x for x in a) ** 0.5
    denom_b = sum(y * y for y in b) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return num / (denom_a * denom_b)
