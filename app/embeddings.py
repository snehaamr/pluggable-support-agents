import httpx

from app.config import Settings
from app.usage import TokenUsage


class OpenAIEmbedder:
    """Embeds text with the same API key and base URL as the chat model."""

    def __init__(self, model_name: str, base_url: str, api_key: str, timeout: float) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._cache: dict[str, list[float]] = {}
        self._usage = TokenUsage()

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._cache]
        if missing:
            for text, vector in zip(missing, self._fetch(missing)):
                self._cache[text] = vector
        return [self._cache[text] for text in texts]

    def take_usage(self) -> TokenUsage:
        taken = self._usage
        self._usage = TokenUsage()
        return taken

    def _fetch(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        raw = body.get("usage") or {}
        prompt = int(raw.get("prompt_tokens") or raw.get("total_tokens") or 0)
        self._usage.add(self.model_name, prompt, 0)
        rows = sorted(body.get("data") or [], key=lambda row: row.get("index", 0))
        if len(rows) != len(texts):
            raise RuntimeError("Embedding response did not match the input.")
        return [list(row["embedding"]) for row in rows]


def build_embedder(settings: Settings) -> OpenAIEmbedder | None:
    if settings.model_provider != "openai" or not settings.model_api_key:
        return None
    return OpenAIEmbedder(
        settings.embedding_model,
        settings.model_base_url,
        settings.model_api_key,
        settings.model_timeout_seconds,
    )
