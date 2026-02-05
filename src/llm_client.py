import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


try:
    from shinka.llm.llm import LLMClient as _ShinkaLLMClient  # type: ignore
except Exception:  # pragma: no cover
    _ShinkaLLMClient = None


@dataclass
class LLMResponse:
    content: str
    cost: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None


class LLMClient:
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        temperatures: float = 0.7,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        self._model_names = model_names or [os.getenv("OPENAI_MODEL", "gpt-4o-mini")]
        self._temperature = temperatures
        self._max_tokens = max_tokens

        if _ShinkaLLMClient is not None:
            self._impl = _ShinkaLLMClient(
                model_names=self._model_names,
                temperatures=temperatures,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            self._impl = None

        self._base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or os.getenv("HYPERBOLIC_BASE_URL")
        )
        self._api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("HYPERBOLIC_API_KEY")
        )

    def query(self, msg: str, system_msg: str = "", **kwargs: Any) -> LLMResponse:
        if self._impl is not None:
            return self._impl.query(msg=msg, system_msg=system_msg, **kwargs)

        if not self._base_url or not self._api_key:
            raise RuntimeError(
                "LLM backend unavailable. Either install/provide ShinkaEvolve (shinka.llm.llm.LLMClient) "
                "or set OPENAI_API_KEY and OPENAI_BASE_URL (or LLM_API_KEY/LLM_BASE_URL) for an OpenAI-compatible endpoint."
            )

        base = self._base_url.rstrip("/")
        url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"

        payload = {
            "model": self._model_names[0],
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": msg},
            ],
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

        try:
            content = raw["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response format: {raw}") from e

        return LLMResponse(content=content, cost=None, raw=raw)
