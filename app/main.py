import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.exc import OperationalError

from app.api import router
from app.config import Settings
from app.db import Base, ensure_auth_columns, make_engine, make_session_factory
from app.seed import seed
from app.wiring import build_runtime


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    supervisor, registry, catalog = build_runtime(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _prepare_database(engine, session_factory)
        yield

    app = FastAPI(title="Pluggable Support Agents", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.supervisor = supervisor
    app.state.registry = registry
    app.state.catalog = catalog
    app.include_router(router)
    return app


def _prepare_database(engine, session_factory) -> None:
    last_error: Exception | None = None
    for attempt in range(15):
        try:
            Base.metadata.create_all(engine)
            ensure_auth_columns(engine)
            with session_factory() as db:
                seed(db)
                db.commit()
            return
        except OperationalError as exc:
            last_error = exc
            time.sleep(1 if attempt else 0)
    raise RuntimeError("Database did not become ready") from last_error


app = create_app()
