"""Structured JSON logging via structlog.

The application emits exactly one JSON record per log call on stdout;
SR-22 forbids `print`, so this module is the single entry point for
runtime observability. Callers are expected to invoke
``configure_logging`` once at startup and then obtain loggers through
``get_logger``.
"""

import logging
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(level: str) -> None:
    """Install the structlog pipeline; safe to call more than once."""
    level_int = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            # set_exc_info lives in structlog.dev since the module split; the
            # processor itself is production-safe (it only flips exc_info=True
            # for .exception()/.critical() calls - no dev-only formatting).
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_int),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a structlog logger, optionally bound to the given name."""
    # structlog.get_logger is typed as -> Any in its stubs; cast restores
    # the concrete Protocol installed via make_filtering_bound_logger above.
    return cast(FilteringBoundLogger, structlog.get_logger(name))
