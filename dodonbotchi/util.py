"""
Module containing various helper functions used throughout DoDonBotchi.
"""
import os
import os.path
import time

from datetime import datetime

SERIAL_LETTERS = 'DPRAEWZ5JNI7LB2UQCHKSOFGMVX3TY'
YEAR_BASE = 2018


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


def generate_serial_number(num):
    """
    Generates a sci-fi-mecha-like serial number for the given integer and
    returns it.
    """
    model_num = '{:03}'.format(num % 1000)
    num *= 10

    model_type = ''
    while num >= len(SERIAL_LETTERS):
        model_type += SERIAL_LETTERS[num % len(SERIAL_LETTERS)]
        num //= len(SERIAL_LETTERS)
    model_type += SERIAL_LETTERS[num]

    if len(model_type) > 5:
        model_prefix = model_type[0] + model_type[1].lower()
        model_type = model_prefix + '-' + model_type[2:]

    return model_type + '-' + model_num


def generate_now_serial_number():
    """
    Generates a sci-fi-mecha-like serial number based on the current date and
    time and returns it.
    """
    now = int(time.time())
    return generate_serial_number(now)
