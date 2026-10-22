"""The fail-closed gateway: screen, mask, forward, restore (docs/GATEWAY.md).

The FastAPI app is imported lazily so that ``import privgate.gateway`` works without the
optional ``gateway`` extra (the request model, policy and screening need only the standard
library and the rest of privgate).
"""

from privgate.gateway.policy import Policy
from privgate.gateway.schema import ChatMessage, ChatRequest, Limits, RequestRejected

__all__ = ["ChatMessage", "ChatRequest", "Limits", "Policy", "RequestRejected", "create_app"]


def create_app(*args, **kwargs):
    """See :func:`privgate.gateway.app.create_app` (imported on first use)."""
    from privgate.gateway.app import create_app as _create_app

    return _create_app(*args, **kwargs)
