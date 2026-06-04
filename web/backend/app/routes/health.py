"""GET /api/health — own status plus a passthrough of the RAG service health."""

from fastapi import APIRouter, Request

from ..rag_client import RagClient
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    rag: RagClient = request.app.state.rag
    info = rag.health()
    if info is None:
        return HealthResponse(rag="down")
    status = info.get("status")
    if status == "ok":
        return HealthResponse(rag="ok", chunk_count=info.get("chunk_count"))
    return HealthResponse(rag="loading")
