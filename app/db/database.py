"""
NewsLens — Database Setup (SQLAlchemy)

Uses synchronous SQLAlchemy with a thread-local session factory.
SQLite for development; swap DATABASE_URL for PostgreSQL in production.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from contextlib import contextmanager
from config import DATABASE_URL

# Enable WAL mode for SQLite (better concurrent read performance)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

if "sqlite" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA synchronous=NORMAL")


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def create_tables() -> None:
    """Create all tables on startup. Safe to call multiple times (IF NOT EXISTS)."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Context-manager style session for use in route handlers."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# FastAPI dependency-injection style
def db_session():
    """Yield a session for FastAPI Depends()."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
