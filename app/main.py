"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.lifespan import lifespan
from app.middleware.errors import register_exception_handlers
from app.routes import cart, chat, checkout, dev, health, partials, session

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="AgenticKapruka",
        description="Agentic shopping assistant for Kapruka",
        lifespan=lifespan,
    )
    # Same-origin HTMX only — no CORSMiddleware (cross-origin disabled by default).

    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(cart.router, prefix="/cart", tags=["cart"])
    app.include_router(checkout.router, prefix="/checkout", tags=["checkout"])
    app.include_router(partials.router, prefix="/partials", tags=["partials"])
    app.include_router(session.router, prefix="/session", tags=["session"])
    app.include_router(dev.router, prefix="/dev", tags=["dev"])

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/chat", status_code=307)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    register_exception_handlers(app)

    return app


app = create_app()
