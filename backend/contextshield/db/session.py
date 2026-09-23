import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base

DEFAULT_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/contextshield"


def get_engine(database_url: str):
    return create_engine(database_url)


def get_default_engine():
    url = os.environ.get("CONTEXTSHIELD_DATABASE_URL", DEFAULT_DATABASE_URL)
    return get_engine(url)


def get_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine):
    Base.metadata.create_all(engine)
