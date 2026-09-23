from fastapi import APIRouter, Depends, HTTPException

from contextshield.api import deps
from contextshield.api.schemas import ConversationCreateIn, ConversationOut
from contextshield.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationOut, status_code=201)
def create_conversation(
    payload: ConversationCreateIn, service: ConversationService = Depends(deps.get_conversation_service)
):
    messages = [{"text": m.text, "author": m.author} for m in payload.messages]
    convo = service.create_conversation(messages, source=payload.source, language=payload.language)
    return convo


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: int, service: ConversationService = Depends(deps.get_conversation_service)):
    convo = service.get_conversation(conversation_id)
    if convo is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    limit: int = 20, offset: int = 0, service: ConversationService = Depends(deps.get_conversation_service)
):
    return service.list_conversations(limit=limit, offset=offset)
