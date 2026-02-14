"""OpenAI-compatible embeddings client for memory indexing."""

from __future__ import annotations

import json
from typing import Iterable

import httpx


class EmbeddingClient:
    def __init__(
        self,
        api_key: str | None,
        api_base: str | None,
        model: str,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        batch_size: int = 8,
        max_input_chars: int | None = 4000,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or ""
        self.api_base = (api_base or "").rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format
        self.batch_size = max(1, batch_size)
        self.max_input_chars = max_input_chars
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_base and self.model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.is_configured():
            return []

        normalized: list[str] = []
        for t in texts:
            if self.max_input_chars is not None:
                normalized.append(t[: self.max_input_chars])
            else:
                normalized.append(t)

        outputs: list[list[float]] = []
        for batch in _batch(normalized, self.batch_size):
            outputs.extend(self._embed_batch(batch))
        return outputs

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        url = f"{self.api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, object] = {
            "model": self.model,
            "input": batch,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        if self.encoding_format is not None:
            payload["encoding_format"] = self.encoding_format

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"Embedding request failed: {resp.status_code} {resp.text}",
                    request=resp.request,
                    response=resp,
                )
            data = resp.json()

        if "data" not in data:
            raise ValueError(f"Unexpected embedding response: {json.dumps(data)[:300]}")

        vectors: list[list[float]] = []
        for item in data["data"]:
            vectors.append(item["embedding"])
        return vectors


def _batch(items: Iterable[str], size: int) -> list[list[str]]:
    batch: list[str] = []
    batches: list[list[str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
