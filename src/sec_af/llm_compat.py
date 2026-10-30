"""Compatibilidad de `.ai()` con proveedores sin `response_format=json_schema`.

DeepSeek solo acepta `{"type": "json_object"}`: con `json_schema` responde
"This response_format type is unavailable now". Si `SEC_AF_JSON_OBJECT_MODE=1`,
se envuelve `litellm.acompletion`/`completion` para degradar a `json_object` e
inyectar el schema como instrucción de sistema. La validación pydantic del SDK
sigue aplicándose a la respuesta.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _degrade(kwargs: dict[str, Any]) -> dict[str, Any]:
    rf = kwargs.get("response_format")
    if not (isinstance(rf, dict) and rf.get("type") == "json_schema"):
        return kwargs
    spec = rf.get("json_schema") or {}
    schema = json.dumps(spec.get("schema", {}), ensure_ascii=False)
    hint = (
        "Responde SOLO con un objeto JSON válido (json) que cumpla exactamente este JSON Schema "
        f"({spec.get('name', 'Output')}); sin texto adicional:\n{schema}"
    )
    messages = list(kwargs.get("messages") or [])
    messages.insert(0, {"role": "system", "content": hint})
    return {**kwargs, "messages": messages, "response_format": {"type": "json_object"}}


def install() -> bool:
    if os.getenv("SEC_AF_JSON_OBJECT_MODE", "").lower() not in {"1", "true", "yes"}:
        return False
    import litellm

    if getattr(litellm, "_secaf_json_object", False):
        return True
    orig_async, orig_sync = litellm.acompletion, litellm.completion

    async def acompletion(*args: Any, **kwargs: Any) -> Any:
        return await orig_async(*args, **_degrade(kwargs))

    def completion(*args: Any, **kwargs: Any) -> Any:
        return orig_sync(*args, **_degrade(kwargs))

    litellm.acompletion, litellm.completion = acompletion, completion
    litellm._secaf_json_object = True
    return True
