"""Memory system for persistent agent memory."""

from datetime import datetime
from pathlib import Path

from nanobot.utils.helpers import ensure_dir


class MemoryStore:
    """Three-layer memory: MEMORY.md (long-term) + daily notes + HISTORY.md (grep log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.daily_dir = ensure_dir(self.memory_dir)

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def append_daily(self, entry: str, date: datetime | None = None) -> None:
        daily = self._daily_file(date)
        with open(daily, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def _daily_file(self, date: datetime | None = None) -> Path:
        dt = date or datetime.now()
        return self.daily_dir / f\"{dt.strftime('%Y-%m-%d')}.md\"

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""
