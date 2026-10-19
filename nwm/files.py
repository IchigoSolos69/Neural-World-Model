"""Atomic replacement tolerant of brief Windows scanner file locks."""

import os
from time import sleep


def replace_file(source, destination):
    """Retain the old destination on failure; retry only transient permission failures."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            sleep(.02 * 2**attempt)
