import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.exc import OperationalError

from app.api import router
from app.config import Settings
from app.db import Base, ensure_auth_columns, make_engine, make_session_factory
from app.seed import seed
from app.services import AsgiAgentCaller, HttpAgentCaller, create_order_app, create_refund_app
from app.wiring import build_runtime


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    order_app = create_order_app(settings)
    refund_app = create_refund_app(settings)
    if settings.agent_transport == "asgi":
        caller = AsgiAgentCaller(
            {
                settings.order_agent_url: order_app,
                settings.refund_agent_url: refund_app,
            }
        )
    elif settings.agent_transport == "http":
        caller = HttpAgentCaller(settings.model_timeout_seconds)
    else:
        raise RuntimeError(f"Unknown AGENT_TRANSPORT '{settings.agent_transport}'")
    supervisor, registry, catalog = build_runtime(settings, caller)

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
    app.state.order_app = order_app
    app.state.refund_app = refund_app
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
