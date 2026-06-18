"""The `datasets` challenge source (staged 5-competition manifest corpus)."""

from __future__ import annotations

import src.tui.challenge_loader as cl


def _mk(root, comp, cat, name, with_manifest=True):
    d = root / comp / cat / name
    d.mkdir(parents=True)
    if with_manifest:
        (d / "manifest.json").write_text("{}")
    return d


def test_load_datasets_only_manifest_bearing(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "_DATASETS_DIR", tmp_path)
    _mk(tmp_path, "oscn_2026", "crypto", "rsb")
    _mk(tmp_path, "rocsc_quals_2026", "web", "Y")
    _mk(tmp_path, "oscn_2026", "pwn", "no-manifest", with_manifest=False)

    entries = cl.load_challenges("datasets")
    ids = sorted(e.challenge_id for e in entries)
    assert ids == [
        "oscn_2026/crypto/rsb",
        "rocsc_quals_2026/web/Y",
    ]  # no-manifest excluded
    assert all(e.source == "datasets" for e in entries)
    assert {e.category for e in entries} == {"crypto", "web"}


def test_datasets_category_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "_DATASETS_DIR", tmp_path)
    _mk(tmp_path, "c1", "crypto", "a")
    _mk(tmp_path, "c1", "web", "b")
    assert cl.get_available_categories("datasets") == ["crypto", "web"]
    assert [e.challenge_id for e in cl.load_challenges("datasets", ["web"])] == [
        "c1/web/b"
    ]
