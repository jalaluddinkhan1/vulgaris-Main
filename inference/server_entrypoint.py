"""Production entrypoint: loads config from environment and starts uvicorn."""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve.logging_setup import setup_logging

logger = setup_logging(os.environ.get("VULGARIS_LOG_LEVEL", "INFO"))


def main():
    import uvicorn
    port = int(os.environ.get("VULGARIS_PORT", "8000"))
    host = os.environ.get("VULGARIS_HOST", "0.0.0.0")
    workers = int(os.environ.get("VULGARIS_WORKERS", "1"))

    logger.info("Starting VULGARIS inference server", extra={
        "ctx_port": port,
        "ctx_host": host,
        "ctx_workers": workers,
    })

    uvicorn.run(
        "inference.server:app",
        host=host,
        port=port,
        workers=workers,
        log_config=None,  # Use our JSON logger
        access_log=False,
    )


if __name__ == "__main__":
    main()
