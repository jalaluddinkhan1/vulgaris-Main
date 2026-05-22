"""JSON structured logging setup for VULGARIS."""
import json
import logging
import sys
import time


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key.startswith("ctx_"):
                log_obj[key[4:]] = val
        return json.dumps(log_obj)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with JSON output to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logging.getLogger("vulgaris")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vulgaris.{name}")
