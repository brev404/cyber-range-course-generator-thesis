import src.services.llm_service as ls
from src.services.codex_model import CodexModel


def test_get_available_providers_includes_codex(monkeypatch):
    monkeypatch.setattr(
        "shutil.which", lambda n: "/usr/bin/x" if n in ("codex", "claude") else None
    )
    assert "codex" in ls.get_available_providers()


def test_get_chat_model_codex_keeps_explicit_gpt(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/x")
    model = ls.get_chat_model(provider="codex", model="gpt-5.5")
    assert isinstance(model, CodexModel)
    assert model.model_name == "gpt-5.5"


def test_get_chat_model_codex_ignores_non_codex_model(monkeypatch):
    """A polluting non-Codex model (e.g. from LLM_DEFAULT_MODEL) is ignored."""
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/x")
    model = ls.get_chat_model(provider="codex", model="claude-sonnet-4-6")
    assert isinstance(model, CodexModel)
    assert model.model_name == "gpt-5.5"


def test_explicit_provider_beats_run_contextvar(monkeypatch):
    """Regression for Bug A: an explicit provider/model (e.g. the judge's
    RANKING_PROVIDER) must win over the ambient _run_* ContextVars that the
    generator sets for the run. Otherwise the refinement re-ranking judges on
    the generator model instead of the configured judge."""
    from src.services.claude_code_model import ClaudeCodeModel

    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/x")
    tok_p = ls._run_provider.set("codex")
    tok_m = ls._run_model.set("gpt-5.5")
    try:
        model = ls.get_chat_model(provider="claude-code", model="claude-haiku-4-5")
        assert isinstance(model, ClaudeCodeModel)
        assert model.model_name == "claude-haiku-4-5"
    finally:
        ls._run_provider.reset(tok_p)
        ls._run_model.reset(tok_m)


def test_run_contextvar_applies_when_no_explicit_provider(monkeypatch):
    """Generation path: with no explicit provider, the run ContextVar still
    selects the generator (so the precedence flip does not break generation)."""
    from src.services.codex_model import CodexModel

    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/x")
    tok_p = ls._run_provider.set("codex")
    tok_m = ls._run_model.set("gpt-5.5")
    try:
        model = ls.get_chat_model()
        assert isinstance(model, CodexModel)
    finally:
        ls._run_provider.reset(tok_p)
        ls._run_model.reset(tok_m)
