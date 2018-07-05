import itertools
import logging as log
import math
import numpy as np
import os
import random

from time import sleep

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from shapely import geometry

from .mame import Ddonpach, get_action_str
from .util import ensure_directories

TRACK_SPAN = 2
MAX_ENTITY_DISTANCE = 32

ENTITY_KEYS = set([
    'ship',
    'bullets',
    'enemies',
    'ownshot',
    'powerup',
    'bonuses',
])

VERT = [0, 1, 2]
HORI = [0, 1, 2]
DIRECTIONS = list()

BOMB_DISTANCE = 10

SCREEN_W = 240
SCREEN_H = 320

HEAT_FILTER = ImageFilter.GaussianBlur(radius=12)


def generate_directions():
    for vert in VERT:
        DIRECTIONS.append(list())
        y_diff = 0
        if vert == 1:
            y_diff = -1
        if vert == 2:
            y_diff = 1

        for hori in HORI:
            x_diff = 0
            if hori == 1:
                x_diff = -1
            if hori == 2:
                x_diff = 1

            if x_diff != 0 and y_diff != 0:
                x_diff *= math.sqrt(2.0)
                y_diff *= math.sqrt(2.0)

            DIRECTIONS[vert].append((x_diff, y_diff))


def flip_xy(state):
    for key, entities in state.items():
        if key not in ENTITY_KEYS:
            continue

        for entity in entities:
            entity['pos_x'], entity['pos_y'] = entity['pos_y'], entity['pos_x']
            entity['siz_x'], entity['siz_y'] = entity['siz_y'], entity['siz_x']


def stabilise(state):
    for key, entities in state.items():
        if key not in ENTITY_KEYS:
            continue

        for entity in entities:
            entity['pos_x'] = entity['pos_x'] + 40 + state['x_off']


def reorient(state):
    for key, entities in state.items():
        if key not in ENTITY_KEYS:
            continue

        for entity in entities:
            # entity['pos_x'] = SCREEN_W - entity['pos_x']
            entity['pos_y'] = SCREEN_H - entity['pos_y']


def ddonpai_loop(ddonpach, draw=False, show=False):
    generate_directions()

    observations = [ddonpach.read_observation()]
    ddonpach.send_action(get_action_str(shot=1))
    step = 1
    while True:
        observation = ddonpach.read_observation()
        for key in ('enemies', 'bullets', 'ownshot', 'powerup', 'bonuses'):
            flip_xy(observation[key])

        observations.append(observation)
        if len(observations) > TRACK_SPAN:
            observations = observations[len(observations) - TRACK_SPAN:]

        ddonpai_step(ddonpach, step, observations[0], observations[-1],
                     draw=(draw or show))
        if show:
            plt.ion()
            plt.show()
            plt.pause(0.001)
            plt.cla()

        step += 1


def entities_heat(draw, entities, colour):
    for entity in entities:
        min_x = entity['pos_x'] - entity['siz_x'] / 2
        min_y = entity['pos_y'] - entity['siz_y'] / 2
        max_x = entity['pos_x'] + entity['siz_x'] / 2
        max_y = entity['pos_y'] + entity['siz_y'] / 2
        ellipse = (min_x, min_y, max_x, max_y)
        draw.ellipse(ellipse, fill=colour)


def enemies_heat(draw, enemies):
    relevant = []
    for enemy in enemies:
        if enemy['siz_x'] != 32 or enemy['siz_y'] != 32:
            relevant.append(enemy)
    return entities_heat(draw, relevant, '#111111')


def state_heat(state):
    img = Image.new('L', (SCREEN_H, SCREEN_H), '#000000')

    draw = ImageDraw.Draw(img)

    for _ in range(1):
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, SCREEN_H - 1, SCREEN_H - 1), outline='#050505')
        entities_heat(draw, state['bullets'], '#101010')
        enemies_heat(draw, state['enemies'])
        img = img.filter(HEAT_FILTER)

    return np.array(img)


class DdonpAi:
    cooling_rate = 8

    def __init__(self, ddonpach, snapshots=False):
        self.ddonpach = ddonpach
        self.heatmap = np.zeros((SCREEN_H, SCREEN_H), dtype=np.int16)
        self.tick = 0

        self.snapshots = snapshots
        self.snap = None

        self.mdp = None

    def cool_down(self):
        self.heatmap -= DdonpAi.cooling_rate
        self.heatmap = self.heatmap.clip(0, 255)

    def heat_up(self, state):
        heat = state_heat(state)
        self.heatmap += heat
        self.heatmap = self.heatmap.clip(0, 255)

    def update_mdp(self, state):
        ship_pos = state['ship'][0]
        ship_pos = np.array((ship_pos['pos_y'], ship_pos['pos_x']))

        centre = np.array((320 - SCREEN_H / 3, SCREEN_H / 2))

        powerup = None
        if state['powerup']:
            powerup = state['powerup'][0]
            powerup = np.array((powerup['pos_y'], powerup['pos_x']))

        def candidate_key(candidate):
            ship_dist = int(np.linalg.norm(ship_pos - candidate))
            if ship_dist == 0:
                ship_dist = 1

            other_dist = int(np.linalg.norm(centre - candidate))
            if powerup is not None:
                other_dist = int(np.linalg.norm(powerup - candidate))

            return ship_dist, other_dist

        min_heat = self.heatmap.min()
        candidates = np.where(self.heatmap == min_heat)
        candidates = [np.array(c) for c in zip(*candidates)]
        candidates = sorted(candidates, key=candidate_key)

        self.mdp = candidates[0]
        self.mdp = (self.mdp[1], 320 - self.mdp[0])

    def move_to_mdp(self, state):
        mdp = np.array(self.mdp)

        ship_pos = state['ship'][0]
        ship_pos = np.array((ship_pos['pos_x'], 320-ship_pos['pos_y']))

        def choice_key(choice):
            direction = np.array(DIRECTIONS[choice[0]][choice[1]])
            moved = ship_pos + direction
            distance = np.linalg.norm(mdp - moved)
            return distance

        choices = list(itertools.product(VERT, HORI))
        choices = sorted(choices, key=choice_key)

        return choices[0]

    def draw_state(self, state, img):
        draw = ImageDraw.Draw(img)

        mdp_rect = (self.mdp[0] - 2, SCREEN_H - self.mdp[1] - 2,
                    self.mdp[0] + 2, SCREEN_H - self.mdp[1] + 2)
        draw.rectangle(mdp_rect, fill='#0044FF')

        ship_rect = (state['ship'][0]['pos_x'], state['ship'][0]['pos_y'])
        ship_rect = (ship_rect[0] - 2, ship_rect[1] - 2,
                     ship_rect[0] + 2, ship_rect[1] - 2)
        draw.rectangle(ship_rect, fill='#00FF44')

    def step(self, state):
        self.cool_down()
        self.heat_up(state)

        self.update_mdp(state)

        vert, hori = self.move_to_mdp(state)

        if self.snapshots:
            heat_img = Image.fromarray(self.heatmap.astype(np.uint8))
            heat_img = heat_img.convert('RGB')

            self.draw_state(state, heat_img)

            self.snap.paste(heat_img,
                            (SCREEN_W, 0, SCREEN_W + SCREEN_H, SCREEN_H))

        action = get_action_str(vert=vert, hori=hori, shot=self.tick % 2)
        return action

    def play(self):
        while True:
            state = self.ddonpach.read_observation()
            flip_xy(state)
            stabilise(state)
            reorient(state)

            if self.snapshots:
                current = self.ddonpach.get_snap()
                expanded = Image.new('RGB', (SCREEN_W + SCREEN_H, SCREEN_H))
                expanded.paste(current)
                self.snap = expanded

            action = self.step(state)
            self.ddonpach.send_action(action)

            if self.snapshots:
                self.snap.save('snp_{:06}.png'.format(self.tick))

            self.tick += 1


def play(ddonpai_dir, snapshots=False):
    ensure_directories(ddonpai_dir)
    generate_directions()
    ddonpai_dir = os.path.abspath(ddonpai_dir)

    ddonpach = Ddonpach('rc')
    ddonpach.inp_dir = ddonpai_dir
    ddonpach.snp_dir = ddonpai_dir

    with ddonpach as mame:
        ddonpai = DdonpAi(mame, snapshots)
        ddonpai.play()
