import json
import os
import random

from pathlib import Path
from time import sleep

from .mame import Ddonpach, get_action_str
from .util import ensure_directories


DIRECTIONS = []

for vert in range(3):
    for hori in range(3):
        if not vert and not hori:
            continue
        DIRECTIONS.append((vert, hori))

assert len(DIRECTIONS) == 8

WINDOW_SIZE = 2

SIG_DEATH = 0
SIG_ABORT = 1
SIG_SUCCESS = 2


class Window:
    size = WINDOW_SIZE
    max_cursor = 8 ** WINDOW_SIZE

    def __init__(self, idx, cursor, actions):
        self.idx = idx
        self.cursor = cursor
        self.actions = actions

    def get_idx_str(self):
        idx_str = '{:09}'.format(self.idx)
        return idx_str

    def get_oct_cursor(self):
        oct_cursor = oct(self.cursor)[2:]
        if len(oct_cursor) < Window.size:
            oct_cursor = ('0' * (Window.size - len(oct_cursor))) + oct_cursor

        return oct_cursor[::-1]

    def get_action(self, step):
        oct_cursor = self.get_oct_cursor()
        action = int(oct_cursor[step])
        vert, hori = self.actions[action]
        #shot = 1
        shot = step % 2
        return get_action_str(vert=vert, hori=hori, shot=shot)


class Bruteforce:
    def __init__(self, cwd):
        cwd = Path(cwd)
        self.sav = cwd / 'sav'
        self.inp = cwd / 'inp'
        self.wnd = cwd / 'wnd'
        snp = cwd / 'snp'
        ensure_directories(*[str(p)
                             for p in [self.sav, self.inp, self.wnd, snp]])

        self.ddonpach = Ddonpach('rc')
        self.ddonpach.inp_dir = str(self.inp)
        self.ddonpach.sav_dir = str(self.sav)
        self.ddonpach.wnd_dir = str(self.wnd)
        self.ddonpach.snp_dir = str(snp)
        self.ddonpach.start_mame()

        self.current_window = None
        self.shadow = 2

    def dec_shadow(self):
        self.shadow -= 1
        if self.shadow < 2:
            self.shadow = 2

    def inc_shadow(self):
        self.shadow += 2

    def save_window(self, window):
        fname = '{} - {}'
        fname = fname.format(window.get_idx_str(), window.cursor)
        fname = self.wnd / fname
        actions = json.dumps(window.actions)
        with open(fname, 'w') as out:
            out.write(actions)

    def load_window(self, fname):
        with open(fname, 'r') as infile:
            actions = infile.read()
            actions = json.loads(actions)
        idx_str, cursor = str(fname.name).split(' - ')
        idx = int(idx_str)
        cursor = int(cursor)
        window = Window(idx, cursor, actions)
        return window

    def init_window(self, idx):
        actions = [*DIRECTIONS]
        random.shuffle(actions)
        window = Window(idx, -1, actions)
        idx_str = window.get_idx_str()
        self.ddonpach.send_save_state(idx_str)
        return window

    def pop_current_window(self):
        windows = sorted([*self.wnd.iterdir()])
        if not windows:
            self.current_window = self.init_window(0)
        else:
            fname = windows[-1]
            self.current_window = self.load_window(fname)
            fname = self.wnd / fname
            os.remove(str(fname))
            self.reset()

    def peek_current_window(self, idx):
        windows = sorted([*self.wnd.iterdir()])
        fname = windows[idx]
        self.current_window = self.load_window(fname)

    def run_current_window(self):
        for step in range(Window.size):
            action = self.current_window.get_action(step)
            self.ddonpach.send_action(action)
            observation = self.ddonpach.read_observation()
            if observation['death']:
                if step == 0:
                    return SIG_ABORT
                return SIG_ABORT
        return SIG_SUCCESS

    def next_window(self):
        self.save_window(self.current_window)
        self.current_window = self.init_window(self.current_window.idx + 1)
        self.dec_shadow()

    def reset(self):
        idx_str = self.current_window.get_idx_str()
        self.ddonpach.send_load_state(idx_str)

    def brute_current_window(self):
        while self.current_window.cursor < Window.max_cursor:
            self.current_window.cursor += 1
            result = self.run_current_window()

            if result == SIG_SUCCESS:
                self.next_window()
            elif result == SIG_DEATH:
                self.reset()

            if result == SIG_ABORT:
                break

        self.inc_shadow()
        return False

    def brute(self):
        self.ddonpach.read_observation()
        self.pop_current_window()
        while not self.brute_current_window():
            for _ in range(self.shadow):
                self.pop_current_window()

    def replay(self):
        self.ddonpach.read_observation()
        idx = 0
        while True:
            self.peek_current_window(idx)
            result = self.run_current_window()
            assert result == SIG_SUCCESS
            idx += 1


def force(cwd):
    bruce = Bruteforce(cwd)
    bruce.brute()


def replay(cwd):
    bruce = Bruteforce(cwd)
    bruce.replay()
