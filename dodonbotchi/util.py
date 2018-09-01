"""
Module containing various helper functions used throughout DoDonBotchi.
"""
import os
import os.path
import time

from datetime import datetime

SERIAL_LETTERS = 'DPRAEWZ5JNI7LB2UQCHKSOFGMVX3TY'
YEAR_BASE = (2018 - 1970) * 365 * 24 * 60 * 60  # Seconds between 1970 and 2018


def ensure_directories(*dirs):
    """
    Goes through each directory in the given list of paths and creates it and
    any super directories needed if required.
    """
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def get_now_string():
    """
    Returns the current time in roughly the ISO8601 format, but with : replaced
    with - to avoid file sytem problems and returns it.
    """
    now = datetime.now()
    now = now.isoformat()
    now = now.replace(':', '-')
    now = now[0:now.rfind('.')]
    return now
