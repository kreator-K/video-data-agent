import json
import re


def parse_json_object(raw: str) -> dict:
    """Parse a JSON object from strict JSON, a fenced response, or surrounding prose."""
    text = raw.strip()

    if "```" in text:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise json.JSONDecodeError("No JSON object found", raw, 0) from None
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected JSON object", text, 0)
    return parsed
