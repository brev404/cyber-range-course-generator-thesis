# ruff: noqa: E402  -- imports below sys.path manipulation and loguru sink
# redirect are intentional (Textual demands clean stdout/stderr before app
# import; loguru's default stderr sink must be removed first).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Textual claims the terminal exclusively.  Any write to stderr/stdout while
# it runs corrupts the display (flickering, overlapping text).  Redirect
# loguru to a rotating file BEFORE importing anything that touches the logger.
from loguru import logger as _logger

_logger.remove()  # drop the default stderr sink
_logger.add(
    Path(__file__).parent.parent / "tui.log",
    level="DEBUG",
    rotation="10 MB",
    retention=3,
    encoding="utf-8",
)

from src.tui.app import (
    ContextCreatorApp,
)  # noqa: E402  # must run after sys.path + loguru setup


def main() -> None:
    ContextCreatorApp().run()


if __name__ == "__main__":
    main()
