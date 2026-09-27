"""REST endpoints for the optional BRAIN connection (sign-in, catalog sync, simulate, mine)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..brainio.brain_api import BrainAuthError, BrainError

router = APIRouter(prefix="/api/brain")


def _svc():
    from ..brainio.service import get_service
    from .app import ws_

    return get_service(ws_())


def _start(kind: str, config: dict) -> dict:
    from .app import mgr

    try:
        return {"ok": True, "id": mgr().start(kind, config)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


class LoginIn(BaseModel):
    email: str = ""
    password: str = ""
    remember: bool = False


class LogoutIn(BaseModel):
    forget: bool = False


class SimulateIn(BaseModel):
    alpha_ids: list[int] = Field(default_factory=list)
    exprs: list[str] = Field(default_factory=list)
    settings: dict | None = None
    check: bool | None = None


class FieldsIn(BaseModel):
    datasets: list[str] = Field(default_factory=list)
    auto: bool = False
    auto_n: int = 12
    region: str | None = None
    delay: int | None = None
    universe: str | None = None
    max_fields_per_dataset: int = 600


class SyncIn(BaseModel):
    limit: int = 300
    stage: str | None = None
    pnl: bool = True


@router.get("/status")
def status() -> dict:
    return _svc().status()


@router.post("/login")
def login(body: LoginIn) -> dict:
    try:
        return _svc().connect(body.email, body.password, body.remember)
    except BrainAuthError as e:
        raise HTTPException(401, str(e)) from None
    except BrainError as e:
        raise HTTPException(502, str(e)) from None


@router.post("/persona/complete")
def persona_complete() -> dict:
    try:
        return _svc().complete_persona()
    except BrainAuthError as e:
        raise HTTPException(401, str(e)) from None


@router.post("/logout")
def logout(body: LogoutIn) -> dict:
    return _svc().disconnect(body.forget)


@router.post("/settings")
def settings(body: dict) -> dict:
    return _svc().update_settings(body)


@router.get("/datasets")
def datasets() -> dict:
    from ..catalog import field_map
    from .app import ws_

    d = ws_().store.kv_get("brain_datasets") or {"datasets": [], "scope": None, "synced": None}
    fm = field_map()
    counts: dict[str, int] = {}
    for f in fm.values():
        if f.get("source_brain"):
            counts[str(f.get("dataset"))] = counts.get(str(f.get("dataset")), 0) + 1
    d["imported_fields"] = counts
    return d


@router.post("/sync/fields")
def sync_fields(body: FieldsIn) -> dict:
    return _start("brain_fields", body.model_dump(exclude_none=True))


@router.post("/sync/alphas")
def sync_alphas(body: SyncIn) -> dict:
    return _start("brain_sync", body.model_dump(exclude_none=True))


@router.post("/simulate")
def simulate(body: SimulateIn) -> dict:
    if not body.alpha_ids and not body.exprs:
        raise HTTPException(400, "Pass alpha ids or expressions to simulate on BRAIN.")
    return _start("brain_sim", body.model_dump(exclude_none=True))


@router.post("/mine")
def mine(config: dict) -> dict:
    return _start("brain_mine", config)


@router.get("/alpha/{alpha_id}")
def alpha(alpha_id: int) -> dict:
    return {"rows": _svc().brain_rows(alpha_id)}


__all__ = ["router"]
