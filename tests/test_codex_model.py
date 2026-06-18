import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

import src.services.codex_model as cm


def _fake_run_writing(content, rc=0, stderr=""):
    """Return a fake subprocess.run that writes `content` to the -o file."""

    def _run(cmd, input=None, capture_output=True, text=True, timeout=None):
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_text(content, encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, rc, stdout="transcript noise", stderr=stderr
        )

    return _run


def test_generate_reads_last_message_file(monkeypatch):
    monkeypatch.setattr(cm, "_is_available", lambda: True)
    monkeypatch.setattr(cm.subprocess, "run", _fake_run_writing("GENERATED COURSE"))
    model = cm.CodexModel()
    result = model._generate([HumanMessage(content="hi")])
    assert result.generations[0].message.content == "GENERATED COURSE"
    assert model.model_name == "gpt-5.5"


def test_generate_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(cm, "_is_available", lambda: True)

    def _run(cmd, input=None, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="auth error")

    monkeypatch.setattr(cm.subprocess, "run", _run)
    with pytest.raises(ValueError, match="exited with code 1"):
        cm.CodexModel()._generate([HumanMessage(content="hi")])


def test_generate_empty_response_raises(monkeypatch):
    monkeypatch.setattr(cm, "_is_available", lambda: True)
    monkeypatch.setattr(cm.subprocess, "run", _fake_run_writing(""))
    with pytest.raises(ValueError, match="empty response"):
        cm.CodexModel()._generate([HumanMessage(content="hi")])


def test_generate_missing_cli_raises(monkeypatch):
    monkeypatch.setattr(cm, "_is_available", lambda: False)
    with pytest.raises(ValueError, match="codex CLI not found"):
        cm.CodexModel()._generate([HumanMessage(content="hi")])


def test_generate_writes_telemetry(monkeypatch, tmp_path):
    from src.services.claude_code_model import set_telemetry_dir

    monkeypatch.setattr(cm, "_is_available", lambda: True)
    monkeypatch.setattr(cm.subprocess, "run", _fake_run_writing("X"))
    set_telemetry_dir(tmp_path)
    try:
        cm.CodexModel()._generate([HumanMessage(content="hi")])
        lines = (
            (tmp_path / "llm_calls.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        assert len(lines) == 1
        assert '"model": "gpt-5.5"' in lines[0]
    finally:
        set_telemetry_dir(None)
