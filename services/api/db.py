from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from services.api.settings import settings


Base = declarative_base()


class VoiceProfile(Base):
    __tablename__ = "voice_profiles"
    id = Column(String, primary_key=True)
    person_name = Column(String, nullable=False)
    source_path = Column(String, nullable=False)
    embedding_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AvatarProfile(Base):
    __tablename__ = "avatar_profiles"
    id = Column(String, primary_key=True)
    person_name = Column(String, nullable=False)
    source_path = Column(String, nullable=False)
    cache_path = Column(String, nullable=False)
    fps = Column(String, default="25")
    created_at = Column(DateTime, default=datetime.utcnow)


class SessionRecord(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    voice_profile_id = Column(String, nullable=False)
    avatar_profile_id = Column(String, nullable=False)
    state = Column(String, default="idle")  # idle, streaming, degraded, closed
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
