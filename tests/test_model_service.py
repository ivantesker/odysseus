"""Model service pure logic — curation, parsing, classification."""
from src.services import model_service as ms


def test_curate_partitions_and_orders():
    models = ["gpt-4o-mini", "gpt-4o", "some-random-model", "o3"]
    curated, extra = ms.curate_models(models, "openai")
    assert "some-random-model" in extra
    # gpt-4o ranks before gpt-4o-mini per the curated priority order.
    assert curated.index("gpt-4o") < curated.index("gpt-4o-mini")


def test_curate_unknown_provider_returns_all():
    assert ms.curate_models(["a", "b"], "nope") == (["a", "b"], [])


def test_curate_openrouter_no_curation():
    assert ms.curate_models(["x", "y"], "openrouter") == (["x", "y"], [])


def test_truthy():
    assert ms.truthy("YES") and ms.truthy("1") and ms.truthy("on")
    assert not ms.truthy("nope") and not ms.truthy("") and not ms.truthy(None)


def test_normalize_endpoint_kind():
    assert ms.normalize_endpoint_kind("API") == "api"
    assert ms.normalize_endpoint_kind("garbage") == "auto"
    assert ms.normalize_endpoint_kind(None) == "auto"


def test_normalize_refresh_mode_proxy_defaults_manual():
    assert ms.normalize_refresh_mode("", "proxy") == "manual"
    assert ms.normalize_refresh_mode("auto", "proxy") == "manual"
    assert ms.normalize_refresh_mode("auto", "local") == "auto"
    assert ms.normalize_refresh_mode("disabled", "local") == "disabled"


def test_parse_positive_int_bounds():
    assert ms.parse_positive_int("5") == 5
    assert ms.parse_positive_int("0", minimum=1) is None
    assert ms.parse_positive_int("999", maximum=60) == 60
    assert ms.parse_positive_int("x") is None


def test_parse_model_list_shapes():
    assert ms.parse_model_list('["a","b","a"]') == ["a", "b"]
    assert ms.parse_model_list("a, b\nc") == ["a", "b", "c"]
    assert ms.parse_model_list(["x", " x ", "y"]) == ["x", "y"]
    assert ms.parse_model_list(None) == []


def test_normalize_and_merge_model_ids():
    assert ms.normalize_model_ids('["a", "b"]') == ["a", "b"]
    assert ms.merge_model_ids(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_visible_models_merges_pinned_and_hides():
    out = ms.visible_models(["a", "b"], hidden_models=["b"], pinned_models=["c"])
    assert out == ["a", "c"]


def test_is_chat_model():
    assert ms.is_chat_model("gpt-4o")
    assert ms.is_chat_model("llama3.1:8b")
    assert not ms.is_chat_model("text-embedding-3-large")
    assert not ms.is_chat_model("whisper-1")
    assert not ms.is_chat_model("dall-e-3")
    assert not ms.is_chat_model("gpt-5.2-codex")
