from collections.abc import Callable
from typing import TypeVar

from openai import OpenAI

from src.model_config import NVIDIA_BASE_URL, api_request_timeout_seconds, nvidia_api_keys


T = TypeVar("T")


def build_nvidia_client(api_key: str | None) -> OpenAI:
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        timeout=api_request_timeout_seconds(),
        max_retries=0,
    )


def build_nvidia_clients() -> list[OpenAI]:
    return [build_nvidia_client(api_key) for api_key in nvidia_api_keys()]


def call_with_client_fallback(
    clients: list[OpenAI],
    operation: Callable[[OpenAI], T],
    retry_message: str,
) -> tuple[T, int]:
    """Run an operation against configured clients, returning its one-based key position."""
    if not clients:
        raise RuntimeError("No NVIDIA API keys are configured.")

    last_error: Exception | None = None
    for index, api_client in enumerate(clients, start=1):
        try:
            return operation(api_client), index
        except Exception as exc:
            last_error = exc
            if index < len(clients):
                print(retry_message)

    raise RuntimeError("All configured NVIDIA API keys failed.") from last_error
