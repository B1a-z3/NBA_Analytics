"""
Central database connection helper. Every layer (ingestion, features, models,
api) imports from here so there's exactly one place that knows how to talk
to Postgres.
"""
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://nba_admin:nba_admin_pw@localhost:5432/nba_analytics",
)

# pool_pre_ping avoids stale-connection errors after the DB idles (common in
# local dev / Docker restarts)
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_engine():
    return engine
