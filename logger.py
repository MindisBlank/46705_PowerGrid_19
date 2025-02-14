import os
from datetime import datetime
from functools import wraps
from loguru import logger
import inspect
import numpy as np

def setup_logger():
    """Sets up the logger to log to a file named with the caller's module name and timestamp."""
    if not os.path.exists("Logs"):
        os.makedirs("Logs")
    # Get the caller's filename from the call stack:
    caller_frame = inspect.stack()[1]
    caller_filename = os.path.basename(caller_frame.filename)
    caller_name = os.path.splitext(caller_filename)[0]
    # Build the log filename using the caller module name:
    log_filename = os.path.join("Logs", f"{caller_name}_{datetime.now().strftime('%d_%m_%y_%H-%M')}.log")
    logger.remove()  # Remove the default handler.
    logger.add(log_filename, level="DEBUG")

def short_repr(obj):
    """
    Returns a short representation of an object.
    For NumPy arrays, returns only shape and dtype.
    """
    if isinstance(obj, np.ndarray):
        return f"array(shape={obj.shape}, dtype={obj.dtype})"
    return repr(obj)

def log_function(func):
    """
    Decorator that logs function entry, arguments, and return value
    only if the function is called with debug=True in its keyword arguments.
    Uses a short representation for large objects.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        is_debug = kwargs.get("debug", False)
        if is_debug:
            #short_args = tuple(short_repr(a) for a in args)
            #short_kwargs = {k: short_repr(v) for k, v in kwargs.items()}
            logger.debug(f"Entering {func.__name__}() with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        if is_debug:
            logger.debug(f"{func.__name__}() returned {result}")
        return result
    return wrapper
