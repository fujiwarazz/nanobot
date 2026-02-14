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
    """Normalized metadata + content payload for a file pending indexing."""

    path: str
    abs_path: Path
    mtime: float
    size: int
    hash: str
    content: str
    source: str


class MemoryIndexManager:
    """Builds and queries a local SQLite memory index with optional hybrid search."""

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
        # `chunks`: canonical chunk storage; `meta`: one-time/full-index markers;
        # `files`: file-level fingerprint cache for incremental sync.
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
                # FTS is optional, so deployments without sqlite fts5 still work.
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(text, content='chunks', content_rowid='id');
                    """
                )
            self._migrate_schema(conn)
            conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Best-effort migration for older local DBs created by previous builds."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        required: dict[str, str] = {
            "source": "TEXT",
            "path": "TEXT",
            "chunk_index": "INTEGER",
            "start_line": "INTEGER",
            "end_line": "INTEGER",
            "text": "TEXT",
            "hash": "TEXT",
            "embedding": "TEXT",
        }
        for col, sql_type in required.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {sql_type}")

        # Old DBs may have no file fingerprint cache.
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

    def start(self) -> None:
        # Optional background watcher for periodic sync.
        if self.config.sync.watch and self._watch_thread is None:
            self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._watch_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=2)

    def sync(self, reason: str = "manual") -> None:
        """Synchronize configured sources into the local index.

        Full reindex happens on first successful run (no meta marker).
        Subsequent runs are incremental: unchanged files are skipped by hash/mtime/size.
        """
        # Keep watch-mode sync quiet in normal terminal output.
        if reason == "watch":
            logger.debug(f"Memory sync: {reason}")
        else:
            logger.info(f"Memory sync: {reason}")
        needs_reindex = True
        meta = self._read_meta()
        if meta:
            # If meta exists, we already had at least one successful full index.
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
        """Search memory chunks by text, then optionally rerank with vectors.

        Returns chunk dicts containing location/text/score (without embedding payload).
        """
        if self.config.sync.on_search:
            self.sync(reason="search")

        if not query.strip():
            return []

        limit = limit or self.config.query.max_results
        results: list[dict[str, Any]] = []

        text_candidates = self._text_search(query, limit * int(self.config.query.hybrid.candidate_multiplier))

        if not self._embedder.is_configured() or not self.config.query.hybrid.enabled:
            # Fall back to text-only retrieval when vector search is unavailable.
            return [self._sanitize_result(c) for c in text_candidates[:limit]]

        # Compute embeddings for query and re-rank
        query_vecs = self._embedder.embed_texts([query])
        if not query_vecs:
            return [self._sanitize_result(c) for c in text_candidates[:limit]]
        qv = query_vecs[0]

        scored: list[tuple[float, dict[str, Any]]] = []
        for cand in text_candidates:
            emb = cand.get("embedding")
            if emb is None:
                # Keep text score when this chunk has no vector embedding.
                score = cand.get("score", 0.0)
            else:
                # Hybrid score: weighted cosine similarity + weighted text score.
                score = (
                    self.config.query.hybrid.vector_weight * _cosine(qv, emb)
                    + self.config.query.hybrid.text_weight * cand.get("score", 0.0)
                )
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [self._sanitize_result(r) for _, r in scored[:limit]]
        return results

    @staticmethod
    def _sanitize_result(row: dict[str, Any]) -> dict[str, Any]:
        """Drop internal heavy fields from tool-facing payloads."""
        out = dict(row)
        out.pop("embedding", None)
        return out

    def get_chunk(self, chunk_id: int) -> dict[str, Any] | None:
        """Fetch a single chunk by primary key."""
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
        }

    def _text_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Run the text stage of retrieval.

        With FTS enabled: use sqlite fts5 + bm25.
        Without FTS: return an unranked subset as a compatibility fallback.
        """
        with sqlite3.connect(self.db_path) as conn:
            if self.config.store.fts:
                try:
                    # bm25 returns lower-is-better. We normalize later into higher-is-better.
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
                except sqlite3.OperationalError:
                    # FTS query parser rejects symbols like "[" from raw chat text.
                    like = f"%{query.replace('[', ' ').replace(']', ' ').strip()}%"
                    rows = conn.execute(
                        """
                        SELECT id, source, path, start_line, end_line, text, embedding, 0.0
                        FROM chunks
                        WHERE text LIKE ?
                        LIMIT ?;
                        """,
                        (like, limit),
                    ).fetchall()
            else:
                # Non-FTS fallback: return deterministic chunk subset without relevance ranking.
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
                    # Convert bm25 (lower is better) to a bounded higher-is-better value.
                    "score": 1.0 / (1.0 + row[7]) if row[7] is not None else 0.0,
                }
            )
        return results

    def _sync_memory_files(self, needs_reindex: bool) -> None:
        """Index workspace-level markdown memory notes under `./memory`."""
        memory_dir = self.workspace_dir / "memory"
        if not memory_dir.exists():
            return
        files = list(memory_dir.glob("*.md"))
        self._sync_files(files, source="memory", needs_reindex=needs_reindex)

    def _sync_session_files(self, needs_reindex: bool) -> None:
        """Index local session transcripts under `~/.nanobot/sessions`."""
        sessions_dir = Path.home() / ".nanobot" / "sessions"
        if not sessions_dir.exists():
            return
        files = list(sessions_dir.glob("*.jsonl"))
        self._sync_files(files, source="sessions", needs_reindex=needs_reindex)

    def _sync_extra_paths(self, needs_reindex: bool) -> None:
        """Index user-configured extra files/directories from config."""
        paths: list[Path] = []
        for p in self.config.extra_paths:
            path = Path(p).expanduser()
            if path.is_dir():
                # Recursively include files under configured directories.
                paths.extend(list(path.glob("**/*")))
            elif path.is_file():
                paths.append(path)
        self._sync_files(paths, source="extra", needs_reindex=needs_reindex)

    def _sync_files(self, files: list[Path], source: str, needs_reindex: bool) -> None:
        """Normalize files -> `MemoryFileEntry`, index changed ones, clean stale ones."""
        entries: list[MemoryFileEntry] = []
        for path in files:
            if not path.is_file():
                continue
            if path.suffix not in (".md", ".txt", ".jsonl") and source == "extra":
                # Keep extra paths constrained to known text-like formats.
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
                # Skip unchanged files for incremental sync.
                continue
            self._index_file(entry)
        # Remove chunks/files that no longer exist in current source set.
        self._delete_stale(source, active)

    def _index_file(self, entry: MemoryFileEntry) -> None:
        """Chunk one file and upsert its chunks + file fingerprint."""
        # Split into overlapping chunks to balance retrieval granularity and context.
        chunks = chunk_text(
            entry.content,
            target_tokens=self.config.chunking.tokens,
            overlap_tokens=self.config.chunking.overlap,
        )
        if not chunks:
            return
        hashes = [
            # Stable per-chunk fingerprint to enable embedding reuse across runs.
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
                # Track file-level metadata for cheap change detection next sync.
                "INSERT OR REPLACE INTO files (path, source, hash, mtime, size) VALUES (?, ?, ?, ?, ?)",
                (entry.path, entry.source, entry.hash, entry.mtime, entry.size),
            )
            if self.config.store.fts:
                # Rebuild FTS shadow table after chunk mutations.
                conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")
            conn.commit()

    def _resolve_embeddings(self, hashes: list[str], texts: list[str]) -> list[list[float]]:
        """Resolve embeddings in hash order using DB cache + batched API calls.

        The return list aligns with `hashes`: each item is the vector for that chunk.
        """
        if not self._embedder.is_configured():
            return []

        cached: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        missing_hashes: list[str] = []

        with sqlite3.connect(self.db_path) as conn:
            for h in hashes:
                # Reuse cached vectors to avoid repeated embedding API calls.
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
            # Only embed chunks that are not already cached.
            embeddings = self._embedder.embed_texts(missing_texts)
            for h, e in zip(missing_hashes, embeddings):
                cached[h] = e

        return [cached.get(h, []) for h in hashes]

    def _file_hash_matches(self, entry: MemoryFileEntry) -> bool:
        """Check whether file fingerprint is unchanged since last index."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT hash, mtime, size FROM files WHERE path = ?",
                (entry.path,),
            ).fetchone()
        if not row:
            return False
        return row[0] == entry.hash and row[1] == entry.mtime and row[2] == entry.size

    def _delete_stale(self, source: str, active_paths: set[str]) -> None:
        """Delete indexed rows for files that disappeared from current source scan."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, path FROM chunks WHERE source = ?",
                (source,),
            ).fetchall()
            stale_ids = [row[0] for row in rows if row[1] not in active_paths]
            if stale_ids:
                # Delete by source to avoid touching chunks from other sources.
                conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in stale_ids])
                conn.executemany("DELETE FROM files WHERE path = ?", [(row[1],) for row in rows if row[1] not in active_paths])
                if self.config.store.fts:
                    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")
                conn.commit()

    def _read_meta(self) -> dict[str, str]:
        """Read simple key-value metadata for index lifecycle."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return {k: v for k, v in rows}

    def _write_meta(self) -> None:
        """Write marker proving at least one full indexing run completed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM meta")
            conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("indexed_at", str(time.time())))
            conn.commit()

    def _watch_loop(self) -> None:
        # Lightweight polling loop guarded by configured intervals/debounce.
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
    """Return SHA-256 hex digest for deterministic content fingerprinting."""
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_session_content(content: str, source: str, path: Path) -> str:
    """Normalize session JSONL into plain `role: content` lines for retrieval.

    Metadata rows are dropped to reduce indexing noise.
    """
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
    """Compute cosine similarity for two vectors with safe zero checks."""
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    denom_a = sum(x * x for x in a) ** 0.5
    denom_b = sum(y * y for y in b) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return num / (denom_a * denom_b)
