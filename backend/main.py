"""
DriveMind development server entry point.

This HTTP surface exists so the backend can be exercised during
development (§29/§273). It is not the product, it has no authentication,
and it must never be bound to a non-loopback interface: `GET /scan`
returns filenames, which is user data.

Run it with:

    python -m backend.main
"""

from fastapi import FastAPI

from backend.api.routes import router
from backend.core.config import get_settings


app = FastAPI(
    title="DriveMind (development interface)",
    description=(
        "Local intelligent disk analysis and cleanup engine. "
        "Development interface only — loopback, unauthenticated, "
        "restricted to DRIVEMIND_ALLOWED_ROOTS."
    ),
    version="0.1.0",
)

app.include_router(router)


def main() -> None:
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        app,
        host=settings.dev_api_host,
        port=settings.dev_api_port,
    )


if __name__ == "__main__":
    main()
