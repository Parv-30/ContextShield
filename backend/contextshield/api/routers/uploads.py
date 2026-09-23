from fastapi import APIRouter, Depends, HTTPException, UploadFile, Form

from contextshield.api import deps
from contextshield.api.schemas import ConversationOut
from contextshield.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/upload", response_model=ConversationOut, status_code=201)
async def upload_conversation(
    file: UploadFile,
    language: str = Form(...),
    service: ConversationService = Depends(deps.get_conversation_service),
):
    if language not in {"en", "hi", "ta"}:
        raise HTTPException(status_code=422, detail="Unsupported language")

    raw_bytes = await file.read()
    lines = [line.strip() for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=422, detail="File contains no messages")

    messages = [{"text": line, "author": None} for line in lines]
    convo = service.create_conversation(messages, source="upload", language=language)
    return convo
