"""Tests for METHODOLOGY_VERSION field in settings and reproducibility export.

Pre-fix: METHODOLOGY_VERSION field does not exist → AttributeError.
Post-fix: field exists, equals 'v4', and is included in export_for_reproducibility().
"""


def test_methodology_version_field_exists():
    """settings must have a METHODOLOGY_VERSION field."""
    from src.config.settings import settings

    assert hasattr(
        settings, "METHODOLOGY_VERSION"
    ), "settings is missing METHODOLOGY_VERSION field"


def test_methodology_version_is_v4_2():
    """METHODOLOGY_VERSION must equal 'v4'."""
    from src.config.settings import settings

    assert (
        settings.METHODOLOGY_VERSION == "v4.2"
    ), f"Expected METHODOLOGY_VERSION='v4', got {settings.METHODOLOGY_VERSION!r}"


def test_methodology_version_in_reproducibility_export():
    """export_for_reproducibility() must include methodology_version='v4'."""
    from src.config.settings import settings

    data = settings.export_for_reproducibility()
    # Must appear somewhere in the top-level dict or under 'settings'
    nested = data.get("settings", {})
    assert (
        nested.get("METHODOLOGY_VERSION") == "v4.2"
        or data.get("methodology_version") == "v4.2"
    ), f"methodology_version not found in reproducibility export: {list(data.keys())}"


def test_token_constants_regression():
    """Token limit constants must be at their v2 bumped floor values."""
    from src.config.settings import settings

    assert (
        settings.CONTENT_GENERATION_MAX_TOKENS >= 14000
    ), f"CONTENT_GENERATION_MAX_TOKENS={settings.CONTENT_GENERATION_MAX_TOKENS} (expected >= 14000)"
    assert (
        settings.CONTENT_GENERATION_SOLVE_MAX_TOKENS >= 6000
    ), f"CONTENT_GENERATION_SOLVE_MAX_TOKENS={settings.CONTENT_GENERATION_SOLVE_MAX_TOKENS} (expected >= 6000)"
