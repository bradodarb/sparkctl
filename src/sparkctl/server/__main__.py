"""`python -m sparkctl.server` — run the unified server in the foreground (uvicorn)."""
import uvicorn

from sparkctl import config
from sparkctl.server.app import create_app


def main():
    uvicorn.run(create_app(), host="0.0.0.0",
                port=config.SERVER.get("port", 8080), log_level="info")


if __name__ == "__main__":
    main()
