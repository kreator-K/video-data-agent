import os


DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_REPORT_MODEL = "meta/llama-3.2-90b-vision-instruct"
DEFAULT_NVIDIA_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_API_REQUEST_TIMEOUT_SECONDS = 30.0


def vision_model() -> str:
    return os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL)


def api_request_timeout_seconds() -> float:
    value = os.getenv("API_REQUEST_TIMEOUT_SECONDS")
    if not value:
        return DEFAULT_API_REQUEST_TIMEOUT_SECONDS
    try:
        return max(float(value), 1.0)
    except ValueError:
        return DEFAULT_API_REQUEST_TIMEOUT_SECONDS


def report_model() -> str:
    return os.getenv("REPORT_MODEL", DEFAULT_REPORT_MODEL)


def nvidia_vision_model() -> str:
    return os.getenv("NVIDIA_VISION_MODEL", DEFAULT_NVIDIA_VISION_MODEL)


def split_api_keys(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",").replace(" ", ",")
    return [key.strip() for key in normalized.split(",") if key.strip()]


def nvidia_api_keys() -> list[str]:
    keys = split_api_keys(os.getenv("NVIDIA_API_KEYS"))
    if keys:
        return keys
    return split_api_keys(os.getenv("NVIDIA_API_KEY"))


def nvidia_api_key() -> str | None:
    keys = nvidia_api_keys()
    return keys[0] if keys else None


def moonshot_api_keys() -> list[str]:
    keys = split_api_keys(os.getenv("MOONSHOT_API_KEYS"))
    if keys:
        return keys
    return split_api_keys(os.getenv("MOONSHOT_API_KEY"))


def moonshot_base_url() -> str:
    return os.getenv("MOONSHOT_BASE_URL", MOONSHOT_BASE_URL)


def vision_base_url() -> str | None:
    return os.getenv("VISION_BASE_URL")


def vision_api_key() -> str | None:
    if os.getenv("VISION_API_KEY"):
        return os.getenv("VISION_API_KEY")
    if vision_base_url() == NVIDIA_BASE_URL or vision_model().startswith(("meta/", "nvidia/")):
        return nvidia_api_key()
    return os.getenv("OPENAI_API_KEY")
