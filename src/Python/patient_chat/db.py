import logging
import os
import iris 

logger = logging.getLogger(__name__)

def _params() -> dict:
    return {
        "hostname": os.getenv("IRIS_HOST", "localhost"),
        "port": int(os.getenv("IRIS_PORT", "9091")),
        "namespace": os.getenv("IRIS_NAMESPACE", "IRISAPP"),
        "username": os.getenv("IRIS_USER", "_SYSTEM"),
        "password": os.getenv("IRIS_PASSWORD", "SYS"),
    }

def connect():
    """Return a fresh DB-API connection to IRIS.

    Caller is responsible for closing it (use a context manager via
    `with closing(connect()) as db: ...` or the helper below).
    """
    p = _params()
    logger.debug("Connecting to IRIS at %s:%s/%s as %s", p["hostname"], p["port"], p["namespace"], p["username"])
    return iris.connect(**p)
