from fastapi import APIRouter, Depends, HTTPException, UploadFile, Form
from pydantic import ValidationError

from contextshield.api import deps
from contextshield.api.schemas import ConversationCreateIn, ConversationOut, MessageIn
from contextshield.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/upload", response_model=ConversationOut, status_code=201)
async def upload_conversation(
    file: UploadFile,
    language: str = Form(...),
    service: ConversationService = Depends(deps.get_conversation_service),
):
    raw_bytes = await file.read()
    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File is not valid UTF-8 text")

    lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=422, detail="File contains no messages")

    try:
        payload = ConversationCreateIn(
            messages=[MessageIn(text=line, author=None) for line in lines],
            source="upload",
            language=language,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    messages = [{"text": m.text, "author": m.author} for m in payload.messages]
    convo = service.create_conversation(messages, source="upload", language=payload.language)
    return convo
