import logging
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx
import uvicorn
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
    if settings.start_agent_services:
        caller = HttpAgentCaller(settings.model_timeout_seconds)
    else:
        caller = AsgiAgentCaller(
            {
                settings.order_agent_url: order_app,
                settings.refund_agent_url: refund_app,
            }
        )
    supervisor, registry, catalog = build_runtime(settings, caller)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _prepare_database(engine, session_factory)
        servers = []
        if settings.start_agent_services:
            servers = [
                _start_service(order_app, settings.order_agent_url),
                _start_service(refund_app, settings.refund_agent_url),
            ]
            _wait_until_ready(settings.order_agent_url)
            _wait_until_ready(settings.refund_agent_url)
        yield
        for server, thread in servers:
            server.should_exit = True
            thread.join(timeout=5)

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


def _start_service(service_app: FastAPI, url: str):
    parsed = urlparse(url)
    config = uvicorn.Config(
        service_app,
        host="0.0.0.0",
        port=parsed.port or 8001,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, name=f"agent-{parsed.port}", daemon=True)
    thread.start()
    return server, thread


def _wait_until_ready(url: str) -> None:
    deadline = time.time() + 15
    health = f"{url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            response = httpx.get(health, timeout=0.5)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"Agent service did not start: {url}")


app = create_app()
