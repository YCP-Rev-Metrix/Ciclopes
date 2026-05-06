from __future__ import annotations

from fastapi import APIRouter

from core.MockDB.mock_db_reader import list_saved_run_names
from src.modules.Query.models import QueryNamesOutput

router = APIRouter(
    prefix="/query",
    tags=["Query"],
)


@router.get("/names", response_model=QueryNamesOutput)
async def query_saved_run_names() -> QueryNamesOutput:
    return QueryNamesOutput(names=list_saved_run_names())
