from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
import json
import os
import types
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import httpx

from privibe.core.logger import logger
from privibe.core.llm.backend.anthropic import AnthropicAdapter
from privibe.core.llm.backend.base import APIAdapter, PreparedRequest
from privibe.core.llm.backend.reasoning_adapter import ReasoningAdapter
from privibe.core.llm.backend.vertex import VertexAnthropicAdapter
from privibe.core.llm.exceptions import BackendErrorBuilder
from privibe.core.llm.message_utils import (
    insert_between_consecutive_assistant_messages,
    merge_consecutive_user_messages,
)
from privibe.core.types import (
    AvailableTool,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StrToolChoice,
)
from privibe.core.utils import async_generator_retry, async_retry

if TYPE_CHECKING:
    from privibe.core.config import ModelConfig, ProviderConfig


class ModelEndpointInfo(NamedTuple):
    """Result of a model-introspection lookup against an OpenAI-compatible server.

    `display_name` is the server-reported model name and is COSMETIC ONLY — it is
    used for display and never affects model matching or config. Either field may
    be None when the server doesn't expose it.
    """

    context_size: int | None
    display_name: str | None


# This is what should be USED for llama.cpp, not anthropic
class OpenAIAdapter(APIAdapter):
    endpoint: ClassVar[str] = "/chat/completions"

    def build_payload(
        self,
        model_name: str,
        converted_messages: list[dict[str, Any]],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
    ) -> dict[str, Any]:
        payload = {
            "model": model_name,
            "messages": converted_messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = [tool.model_dump(exclude_none=True) for tool in tools]
        if tool_choice:
            payload["tool_choice"] = (
                tool_choice
                if isinstance(tool_choice, str)
                else tool_choice.model_dump()
            )
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return payload

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _reasoning_to_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        # When the provider's reasoning field has a different name from
        # privibe's internal `reasoning_content`, rename it on outbound.
        #
        # NOTE — a prior version of this comment claimed sending
        # `reasoning_content` is what keeps the llama.cpp KV cache
        # aligned across multi-turn requests because the server's chat
        # template re-renders {content, reasoning_content} into the same
        # `<think>X</think>Y` byte sequence the model originally emitted.
        # That is wrong for Qwen 3.5 / 3.6: their stock chat template
        # strips `<think>...</think>` from every assistant turn except
        # the latest one (gate: `loop.index0 > ns.last_query_index`).
        # The cache-alignment fix for those models lives in llama.cpp
        # itself (run llama-server with `--chat-template-file PATH`
        # pointed at a copy of the model's chat template where that
        # gate has been replaced by `true`).  Sending `reasoning_content`
        # is still the right thing to do — it's preserved on the latest
        # turn and is a no-op for providers that ignore it — it's just
        # not the load-bearing fix it was advertised as.  See
        # ~/Documents/ContextFiles/context-disk-cache-eviction.md and
        # kv-cache-conversation-object-design.md for the full trace.
        if field_name != "reasoning_content" and "reasoning_content" in msg_dict:
            msg_dict[field_name] = msg_dict.pop("reasoning_content")
        return msg_dict

    def _reasoning_from_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and field_name in msg_dict:
            msg_dict["reasoning_content"] = msg_dict.pop(field_name)
        return msg_dict

    def prepare_request(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        messages: Sequence[LLMMessage],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        enable_streaming: bool,
        provider: ProviderConfig,
        api_key: str | None = None,
        thinking: str = "off",
    ) -> PreparedRequest:
        merged_messages = insert_between_consecutive_assistant_messages(
            merge_consecutive_user_messages(messages)
        )
        field_name = provider.reasoning_field_name
        converted_messages = [
            self._reasoning_to_api(
                msg.model_dump(
                    exclude_none=True,
                    exclude={"message_id", "reasoning_message_id", "injected"},
                ),
                field_name,
            )
            for msg in merged_messages
        ]

        payload = self.build_payload(
            model_name, converted_messages, temperature, tools, max_tokens, tool_choice
        )

        if enable_streaming:
            payload["stream"] = True
            stream_options: dict[str, Any] = {"include_usage": True}
            if getattr(provider, "stream_tool_calls", False):
                stream_options["stream_tool_calls"] = True
            payload["stream_options"] = stream_options

        headers = self.build_headers(api_key)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        # DEBUG LLM COMMUNICATIONS
        # Mirrors the dump in AnthropicAdapter so llamacpp / vLLM /
        # OpenAI-compatible providers also produce a per-call payload
        # snapshot in ./debug/. Filename includes session_id + kind so
        # warmup vs. real-turn dumps don't collide.
        from privibe.core.llm.backend._debug_dump import (
            dump_payload as _dbg_dump_payload,
            is_enabled as _dbg_enabled,
        )
        if _dbg_enabled():
            _dbg_dump_payload(body, payload)

        return PreparedRequest(self.endpoint, headers, body)

    def _parse_message(
        self, data: dict[str, Any], field_name: str
    ) -> LLMMessage | None:
        if data.get("choices"):
            choice = data["choices"][0]
            if "message" in choice:
                msg_dict = self._reasoning_from_api(choice["message"], field_name)
                return LLMMessage.model_validate(msg_dict)
            if "delta" in choice:
                msg_dict = self._reasoning_from_api(choice["delta"], field_name)
                return LLMMessage.model_validate(msg_dict)
            raise ValueError("Invalid response data: missing message or delta")

        if "message" in data:
            msg_dict = self._reasoning_from_api(data["message"], field_name)
            return LLMMessage.model_validate(msg_dict)
        if "delta" in data:
            msg_dict = self._reasoning_from_api(data["delta"], field_name)
            return LLMMessage.model_validate(msg_dict)

        return None

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk:
        message = self._parse_message(data, provider.reasoning_field_name)
        if message is None:
            message = LLMMessage(role=Role.assistant, content="")

        usage_data = data.get("usage") or {}
        # llama.cpp reports generation speed under a non-OpenAI "timings" object
        # (top-level in non-stream responses, in the final chunk when streaming).
        # Other providers omit it; .get keeps that path graceful.
        timings = data.get("timings") or {}
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            tokens_per_second=timings.get("predicted_per_second"),
            prompt_tokens_per_second=timings.get("prompt_per_second"),
        )

        return LLMChunk(message=message, usage=usage, served_model=data.get("model"))


ADAPTERS: dict[str, APIAdapter] = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "vertex-anthropic": VertexAnthropicAdapter(),
    "reasoning": ReasoningAdapter(),
}


class GenericBackend:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        provider: ProviderConfig,
        timeout: float = 720.0,
    ) -> None:
        """Initialize the backend.

        Args:
            client: Optional httpx client to use. If not provided, one will be created.
        """
        self._client = client
        self._owns_client = client is None
        self._provider = provider
        self._timeout = timeout

    async def __aenter__(self) -> GenericBackend:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            self._owns_client = True
        return self._client

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> LLMChunk:
        api_key = (
            os.getenv(self._provider.api_key_env_var)
            if self._provider.api_key_env_var
            else None
        )

        api_style = getattr(self._provider, "api_style", "openai")
        adapter = ADAPTERS[api_style]

        req = adapter.prepare_request(
            model_name=model.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=False,
            provider=self._provider,
            api_key=api_key,
            thinking=model.thinking,
        )

        headers = req.headers
        if extra_headers:
            headers.update(extra_headers)

        base = req.base_url or self._provider.api_base
        url = f"{base}{req.endpoint}"

        try:
            res_data, _ = await self._make_request(url, req.body, headers)
            return adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        api_key = (
            os.getenv(self._provider.api_key_env_var)
            if self._provider.api_key_env_var
            else None
        )

        api_style = getattr(self._provider, "api_style", "openai")
        adapter = ADAPTERS[api_style]

        req = adapter.prepare_request(
            model_name=model.name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            enable_streaming=True,
            provider=self._provider,
            api_key=api_key,
            thinking=model.thinking,
        )

        headers = req.headers
        if extra_headers:
            headers.update(extra_headers)

        base = req.base_url or self._provider.api_base
        url = f"{base}{req.endpoint}"

        try:
            async for res_data in self._make_streaming_request(url, req.body, headers):
                yield adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    class HTTPResponse(NamedTuple):
        data: dict[str, Any]
        headers: dict[str, str]

    @async_retry(tries=3)
    async def _make_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> HTTPResponse:
        client = self._get_client()
        response = await client.post(url, content=data, headers=headers)
        response.raise_for_status()

        response_headers = dict(response.headers.items())
        response_body = response.json()
        return self.HTTPResponse(response_body, response_headers)

    @async_generator_retry(tries=3)
    async def _make_streaming_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> AsyncGenerator[dict[str, Any]]:
        client = self._get_client()
        async with client.stream(
            method="POST", url=url, content=data, headers=headers
        ) as response:
            if not response.is_success:
                await response.aread()
            response.raise_for_status()
            DELIM_CHAR = ":"
            async for line in response.aiter_lines():
                if line.strip() == "":
                    continue

                # SSE comment line (starts with ':') — ignore per the SSE
                # spec. llama.cpp sends ":\n\n" as a keep-alive ping during
                # long prefills (server flag --sse-ping-interval, default
                # 30s) so proxies / undici don't drop the connection. The
                # line arrives here as a bare ":" with no "key: value", so
                # it must be skipped before the format check below.
                if line.startswith(DELIM_CHAR):
                    continue

                if f"{DELIM_CHAR} " not in line:
                    raise ValueError(
                        f"Stream chunk improperly formatted. "
                        f"Expected `key{DELIM_CHAR} value`, received `{line}`"
                    )
                delim_index = line.find(DELIM_CHAR)
                key = line[0:delim_index]
                value = line[delim_index + 2 :]

                if key != "data":
                    # This might be the case with openrouter, so we just ignore it
                    continue
                if value == "[DONE]":
                    return
                yield json.loads(value.strip())

    async def count_tokens(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float = 0.0,
        tools: list[AvailableTool] | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> int:
        probe_messages = list(messages)
        if not probe_messages or probe_messages[-1].role != Role.user:
            probe_messages.append(LLMMessage(role=Role.user, content=""))

        result = await self.complete(
            model=model,
            messages=probe_messages,
            temperature=temperature,
            tools=tools,
            max_tokens=16,  # Minimal amount for openrouter with openai models
            tool_choice=tool_choice,
            extra_headers=extra_headers,
        )
        if result.usage is None:
            raise ValueError("Missing usage in non streaming completion")

        return result.usage.prompt_tokens

    async def fetch_model_endpoint_info(self, model_name: str) -> ModelEndpointInfo:
        """Detect the model's context window size and the server-reported name.

        The name is cosmetic (display-only); it never affects matching or config.
        Both values come from the same two endpoints, so we fetch once:
        1. GET {server_root}/props  — llama.cpp exposes default_generation_settings.n_ctx
           (the actual loaded context, often not in /v1/models) and model_alias (name).
        2. GET {api_base}/models   — standard OpenAI-compatible endpoint; checks meta.n_ctx,
           meta.n_ctx_train (llama.cpp) and context_length / max_model_len etc. (vLLM,
           LM Studio); the name is the matched entry's "id".

        Returns ModelEndpointInfo(context_size, display_name); either may be None.
        """
        api_key = (
            os.getenv(self._provider.api_key_env_var)
            if self._provider.api_key_env_var
            else None
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        client = self._get_client()
        base = self._provider.api_base.rstrip("/")
        # Cosmetic name; populated from whichever endpoint reports it first.
        detected_name: str | None = None

        # ── 1. Try /props (llama.cpp server root endpoint) ────────────────────
        # Strip trailing /v1 to reach the server root.
        server_root = base[:-3] if base.endswith("/v1") else base
        props_url = f"{server_root}/props"
        try:
            response = await client.get(props_url, headers=headers, timeout=5.0)
            response.raise_for_status()
            props = response.json()
            detected_name = props.get("model_alias") or detected_name
            n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            if n_ctx:
                logger.info(
                    "fetch_model_endpoint_info: found n_ctx=%s in /props for model '%s'",
                    n_ctx,
                    model_name,
                )
                return ModelEndpointInfo(int(n_ctx), detected_name)
            logger.info(
                "fetch_model_endpoint_info: /props reachable at %s but no n_ctx found, "
                "falling back to /models",
                props_url,
            )
        except Exception as exc:
            logger.info(
                "fetch_model_endpoint_info: /props not available at %s (%s), trying /models",
                props_url,
                exc,
            )

        # ── 2. Fall back to /v1/models ────────────────────────────────────────
        models_url = f"{base}/models"
        try:
            response = await client.get(models_url, headers=headers, timeout=5.0)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.info("fetch_model_endpoint_info: GET %s failed: %s", models_url, exc)
            return ModelEndpointInfo(None, detected_name)

        models: list[dict[str, Any]] = data.get("data", [])
        if not models:
            logger.info(
                "fetch_model_endpoint_info: /v1/models returned empty list from %s",
                models_url,
            )
            return ModelEndpointInfo(None, detected_name)

        # Find model by exact id match; fall back to the only model if there's just one.
        target: dict[str, Any] | None = None
        for m in models:
            if m.get("id") == model_name:
                target = m
                break
        if target is None and len(models) == 1:
            logger.info(
                "fetch_model_endpoint_info: no exact match for '%s', using only model '%s'",
                model_name,
                models[0].get("id"),
            )
            target = models[0]
        if target is None:
            logger.info(
                "fetch_model_endpoint_info: model '%s' not found among %d models at %s",
                model_name,
                len(models),
                models_url,
            )
            return ModelEndpointInfo(None, detected_name)

        # The matched entry's id is the server's name for it (cosmetic only).
        detected_name = target.get("id") or detected_name

        # llama.cpp exposes context info inside a "meta" object.
        # Prefer n_ctx (actual loaded context) over n_ctx_train (training context).
        meta = target.get("meta", {})
        for field in ("n_ctx", "n_ctx_train"):
            if val := meta.get(field):
                logger.info(
                    "fetch_model_endpoint_info: found meta.%s=%s for model '%s'",
                    field,
                    val,
                    model_name,
                )
                return ModelEndpointInfo(int(val), detected_name)

        # Generic OpenAI-compatible servers (vLLM, LM Studio, etc.)
        for field in ("context_length", "max_context_length", "context_window", "max_model_len"):
            if val := target.get(field):
                logger.info(
                    "fetch_model_endpoint_info: found %s=%s for model '%s'",
                    field,
                    val,
                    model_name,
                )
                return ModelEndpointInfo(int(val), detected_name)

        logger.info(
            "fetch_model_endpoint_info: no context-size field found in model entry for '%s'. "
            "Keys present: %s",
            model_name,
            list(target.keys()),
        )
        return ModelEndpointInfo(None, detected_name)

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None
