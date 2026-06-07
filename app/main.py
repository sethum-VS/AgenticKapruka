"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.lifespan import lifespan
from app.routes import cart, chat, checkout, health, partials


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

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/chat", status_code=307)

    return app


app = create_app()
