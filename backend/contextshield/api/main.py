from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextshield.api.routers import conversations

app = FastAPI(title="ContextShield API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversations.router)
