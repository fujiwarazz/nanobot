"""Tools for memory retrieval."""

from __future__ import annotations

import json

from nanobot.agent.tools.base import Tool
from nanobot.memory import MemoryIndexManager


class MemorySearchTool(Tool):
    name = "memory_search"
    description = "Search indexed memory (MEMORY.md, daily notes, sessions) and return relevant snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, memory_index: MemoryIndexManager) -> None:
        self.memory_index = memory_index

    async def execute(self, query: str, limit: int = 5) -> str:
        results = self.memory_index.search(query, limit=limit)
        return json.dumps(results, ensure_ascii=False)


class MemoryGetTool(Tool):
    name = "memory_get"
    description = "Fetch a memory chunk by id."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    }

    def __init__(self, memory_index: MemoryIndexManager) -> None:
        self.memory_index = memory_index

    async def execute(self, id: int) -> str:
        chunk = self.memory_index.get_chunk(id)
        return json.dumps(chunk or {}, ensure_ascii=False)
