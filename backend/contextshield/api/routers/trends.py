from fastapi import APIRouter, Depends

from contextshield.api import deps
from contextshield.api.schemas import TrendSnapshotOut
from contextshield.db.models import TrendSnapshot
from contextshield.services.trend_service import recompute_trends as run_recompute

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("", response_model=list[TrendSnapshotOut])
def get_trends(
    start: str | None = None,
    end: str | None = None,
    category: str | None = None,
    language: str | None = None,
    session=Depends(deps.get_db_session),
):
    query = session.query(TrendSnapshot)
    if start is not None:
        query = query.filter(TrendSnapshot.time_bucket >= start)
    if end is not None:
        query = query.filter(TrendSnapshot.time_bucket <= end)
    if category is not None:
        query = query.filter(TrendSnapshot.category == category)
    if language is not None:
        query = query.filter(TrendSnapshot.language == language)
    return query.all()


@router.post("/recompute")
def recompute_trends(session=Depends(deps.get_db_session)):
    count = run_recompute(session)
    return {"snapshots_created": count}
