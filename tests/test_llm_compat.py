from sec_af.llm_compat import _degrade


def test_degrada_json_schema_a_json_object() -> None:
    kw = {"messages": [{"role": "user", "content": "x"}],
          "response_format": {"type": "json_schema", "json_schema": {"name": "F", "schema": {"type": "object"}}}}
    out = _degrade(kw)
    assert out["response_format"] == {"type": "json_object"}
    assert out["messages"][0]["role"] == "system" and '"type": "object"' in out["messages"][0]["content"]
    assert out["messages"][1] == {"role": "user", "content": "x"}


def test_no_toca_otras_llamadas() -> None:
    kw = {"messages": [], "response_format": {"type": "json_object"}}
    assert _degrade(kw) is kw
