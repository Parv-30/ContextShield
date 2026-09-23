from fastapi import APIRouter, Depends, HTTPException

from contextshield.api import deps
from contextshield.api.schemas import TrendSnapshotOut
from contextshield.db.models import TrendSnapshot

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
def recompute_trends():
    raise HTTPException(status_code=501, detail="Not implemented yet")
