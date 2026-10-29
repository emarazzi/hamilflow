"""Logging set-up for the jobflow jobs, the one place where hamilflow is the application."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

# same as the examples' logging.basicConfig (see examples/README.md)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


@contextmanager
def job_logging() -> Iterator[None]:
    """Print hamilflow's messages (INFO and above) and captured warnings to stdout, the job's
    output file, for the duration of a job: ``with job_logging(): <job body>``.

    Does nothing if the root logger or the ``hamilflow`` logger already has a handler (the
    caller's own configuration, or an enclosing job), so it never overrides what the user set
    up. The handler goes on the ``hamilflow`` and ``py.warnings`` loggers instead of the root
    one: jobflow's ``initialize_logger`` (``run_locally``) gives the ``jobflow`` logger its own
    stdout handler and leaves it propagating to the root, so a root handler would print every
    jobflow message twice. On exit, also on an exception, the handlers, the ``hamilflow``
    level and the ``logging.captureWarnings`` state are put back as they were, so nothing
    leaks into the rest of the process (e.g. the next job, or a later ``basicConfig``).
    """
    package = logging.getLogger("hamilflow")
    if logging.getLogger().handlers or package.handlers:
        yield
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    warnings_logger = logging.getLogger("py.warnings")
    old_level = package.level
    # captureWarnings(True) stores the original showwarning there; None means it is off
    # (private, hence getattr: if it ever goes, capture counts as off, the usual case)
    capture_was_on = getattr(logging, "_warnings_showwarning", None) is not None
    package.addHandler(handler)
    if old_level == logging.NOTSET:
        package.setLevel(logging.INFO)
    add_to_warnings = not warnings_logger.handlers
    if add_to_warnings:
        warnings_logger.addHandler(handler)
    if not capture_was_on:
        logging.captureWarnings(True)
    try:
        yield
    finally:
        if not capture_was_on:
            logging.captureWarnings(False)
        if add_to_warnings:
            warnings_logger.removeHandler(handler)
        package.setLevel(old_level)
        package.removeHandler(handler)
