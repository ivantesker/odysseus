"""FastAPI dependencies that decouple route handlers from request.state.

Most routes today read the user via ``effective_user(request)`` /
``require_user(request)``, which forces every handler to accept the raw
``Request`` and reach into ``request.state``. That couples domain logic to the
web layer and makes handlers awkward to unit-test.

These thin dependencies wrap the existing auth helpers so a handler can instead
declare what it needs:

    from src.dependencies import RequiredUser

    @router.get("/api/v1/things")
    def list_things(user: str = RequiredUser):
        return service.list_things(owner=user)

Behaviour is identical to the helper functions — this only moves the
``request.state`` access out of the handler body and into the dependency, which
is the seam a service layer can sit behind.
"""

from typing import Optional

from fastapi import Depends, Request

from src.auth_helpers import effective_user, get_current_user, require_user


def _current_user(request: Request) -> str | None:
    return get_current_user(request)


def _effective_user(request: Request) -> str | None:
    return effective_user(request)


def _required_user(request: Request) -> str:
    return require_user(request)


# Ready-to-use Depends() markers for route signatures.
CurrentUser = Depends(_current_user)      # Optional[str]; None when unauthenticated
EffectiveUser = Depends(_effective_user)  # real owner for attribution/ownership
RequiredUser = Depends(_required_user)    # str; raises 401 when unauthenticated
