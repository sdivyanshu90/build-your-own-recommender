"""PII-safe structured logging configuration."""

import logging

from pythonjsonlogger.json import JsonFormatter


def configure_logging(production: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO if production else logging.DEBUG)
