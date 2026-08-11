from contextlib import contextmanager
from contextvars import ContextVar

# module-level context variable
_config_ctx: ContextVar[dict] = ContextVar("config")


@contextmanager
def config_context(config: dict):
    token = _config_ctx.set(config)
    try:
        yield
    finally:
        _config_ctx.reset(token)


def get_current_config() -> dict:
    """Return the current config in context."""
    return _config_ctx.get()
