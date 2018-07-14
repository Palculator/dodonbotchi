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

MOVE = {
    0: 0,
    -1: 1,
    1: 2,
}
VERT = [0, 1, 2]
HORI = [0, 1, 2]
DIRECTIONS = list()

BOMB_DISTANCE = 10

SCREEN_W = 240
SCREEN_H = 320

HEAT_FILTER = ImageFilter.GaussianBlur(radius=8)


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
            entity['pos_y'] = entity['pos_y'] + 40 + state['x_off']


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


def entities_heat(draw, entities, colour, shrink):
    for entity in entities:
        if entity['id'] != 0:
            min_x = entity['pos_x'] - round(entity['siz_x'] / shrink)
            min_y = entity['pos_y'] - round(entity['siz_y'] / shrink)
            max_x = entity['pos_x'] + round(entity['siz_x'] / shrink)
            max_y = entity['pos_y'] + round(entity['siz_y'] / shrink)
            ellipse = (min_x, min_y, max_x, max_y)
            draw.ellipse(ellipse, fill=colour)


def enemies_heat(enemies):
    img = Image.new('L', (SCREEN_H, SCREEN_H), '#000000')
    draw = ImageDraw.Draw(img)

    relevant = []
    for enemy in enemies:
        if enemy['siz_x'] != 32 or enemy['siz_y'] != 32:
            relevant.append(enemy)

    shrink = 2

    for _ in range(3):
        img = img.filter(HEAT_FILTER)
        draw = ImageDraw.Draw(img)
        entities_heat(draw, relevant, '#101010', shrink)

        shrink *= 2

    return np.array(img)


def bullets_heat(last_bullets, bullets):
    img = Image.new('L', (SCREEN_H, SCREEN_H), '#000000')
    draw = ImageDraw.Draw(img)

    projected = []

    for idx, bul in enumerate(bullets):
        if idx >= len(last_bullets):
            break
        if bul['id'] == 0:
            continue
        if last_bullets[idx]['id'] == bul['id']:
            for mul in range(1, 3):
                proj = dict(bul)
                traj = (bul['pos_x'] - last_bullets[idx]['pos_x'],
                        bul['pos_y'] - last_bullets[idx]['pos_y'])
                proj['pos_x'] += traj[0] * mul
                proj['pos_y'] += traj[1] * mul

                projected.append(proj)

    bullets += projected

    shrink = 2

    for _ in range(4):
        img = img.filter(HEAT_FILTER)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, SCREEN_H - 1, SCREEN_H - 1), outline='#FFFFFF')
        entities_heat(draw, bullets, '#999999', shrink)

        shrink *= 2

    return np.array(img)


def state_heat(last_state, state):
    enemies = enemies_heat(state['enemies'])
    bullets = bullets_heat(last_state['bullets'], state['bullets'])

    enemies += bullets
    enemies = np.clip(enemies, 0, 255)

    return enemies


def distance(a_pos, b_pos):
    diff = (a_pos[0] - b_pos[0],
            a_pos[1] - b_pos[1])
    return math.sqrt(diff[0] ** 2 + diff[1] ** 2)


class DdonpAi:
    cooling_rate = 32

    def __init__(self, ddonpach, snapshots=False):
        self.ddonpach = ddonpach
        self.heatmap = np.zeros((SCREEN_H, SCREEN_H), dtype=np.int16)
        self.tick = 0

        self.snapshots = snapshots
        self.snap = None

        self.mdp = None

        self.last_state = None

    def cool_down(self):
        self.heatmap[:, :] = 0
        return
        self.heatmap -= DdonpAi.cooling_rate
        self.heatmap = self.heatmap.clip(0, 255)

    def heat_up(self, state):
        heat = state_heat(self.last_state, state)
        self.heatmap += heat
        self.heatmap = self.heatmap.clip(0, 255)

    def update_mdp(self, state):
        ship_pos = state['ship'][0]
        ship_pos = (ship_pos['pos_x'], ship_pos['pos_y'])

        centre = (SCREEN_H // 2, 320 - SCREEN_H // 3)

        powerup = None
        if state['powerup']:
            powerup = state['powerup'][0]
            powerup = (powerup['pos_x'], powerup['pos_y'])

        def candidate_key(candidate):
            ship_dist = (ship_pos[0] - candidate[0],
                         ship_pos[1] - candidate[1])
            ship_dist = ship_dist[0] ** 2 + ship_dist[1] ** 2
            if ship_dist == 0:
                ship_dist = 320 ** 2

            other_dist = (centre[0] - candidate[0],

                          centre[1] - candidate[1])

            if powerup is not None:
                other_dist = (powerup[0] - candidate[1],
                              powerup[1] - candidate[0])

            other_dist = other_dist[0] ** 2 + other_dist[1] ** 2

            return other_dist, ship_dist

        min_heat = self.heatmap.min()
        candidates = np.where(self.heatmap == min_heat)
        candidates = zip(*candidates)
        candidates = sorted(candidates, key=candidate_key)

        self.mdp = candidates[0]
        self.mdp = (self.mdp[1], self.mdp[0])

    def get_ship_neighbours(self, state, window):
        ship_pos = state['ship'][0]
        ship_pos = (ship_pos['pos_x'], ship_pos['pos_y'])

        neighbours = (
            ship_pos[0] - window,
            ship_pos[0] + window + 1,
            ship_pos[1] - window,
            ship_pos[1] + window + 1,
        )
        neighbours = tuple(np.clip(neighbours, 0, SCREEN_H))
        min_x = neighbours[0]
        max_x = neighbours[1]
        min_y = neighbours[2]
        max_y = neighbours[3]

        neighbours = self.heatmap[min_y:max_y,
                                  min_x:max_x]

        return neighbours, (min_x, min_y, max_x, max_y)

    def get_viable_movements(self, state):
        ship_pos = state['ship'][0]
        ship_pos = (ship_pos['pos_x'], ship_pos['pos_y'])

        neighbours, box = self.get_ship_neighbours(state, 4)

        min_x = box[0] - ship_pos[0]
        min_y = box[1] - ship_pos[1]
        max_x = box[2] - ship_pos[0]
        max_y = box[3] - ship_pos[1]

        neighbours_min = neighbours.min()

        ret = set()
        viable = neighbours == neighbours_min

        for cur_x in range(min_x, max_x):
            for cur_y in range(min_y, max_y):
                if viable[cur_y - min_y, cur_x - min_x]:
                    vert = np.sign(cur_x)
                    hori = np.sign(cur_y)
                    ret.add((vert, hori))
        return list(ret)

    def move_to_mdp(self, state):
        mdp = np.array(self.mdp)

        ship_pos = state['ship'][0]
        ship_pos = np.array((ship_pos['pos_x'], ship_pos['pos_y']))

        def choice_key(choice):
            choice = np.array(choice)
            moved = ship_pos + choice
            dist = np.linalg.norm(mdp - moved)
            return dist

        choices = self.get_viable_movements(state)
        choices = sorted(choices, key=choice_key)
        vert, hori = choices[0]
        vert, hori = MOVE[vert], MOVE[hori]

        return vert, hori

    def draw_state(self, state, img):
        pass

    def step(self, state):
        self.cool_down()
        self.heat_up(state)
        self.update_mdp(state)

        ship = state['ship'][0]
        ship = (ship['pos_x'], ship['pos_y'])
        if distance(ship, self.mdp) > 4:
            vert, hori = self.move_to_mdp(state)
        else:
            vert, hori = 0, 0

        if self.snapshots:
            ship = state['ship'][0]
            heat_img = self.heatmap.astype(np.uint8)
            heat_img[ship['pos_y'], ship['pos_x']] = 255
            if self.mdp:
                heat_img[self.mdp[1], self.mdp[0]] = 255

            heat_img = Image.fromarray(heat_img)
            heat_img = heat_img.convert('RGB')

            self.draw_state(state, heat_img)
            heat_img = heat_img.transpose(Image.ROTATE_90)

            self.snap.paste(heat_img,
                            (SCREEN_W, 0, SCREEN_W + SCREEN_H, SCREEN_H))

        action = get_action_str(vert=vert, hori=hori, shot=self.tick % 2)
        self.last_state = state
        return action

    def play(self):
        self.last_state = self.ddonpach.read_observation()
        self.ddonpach.send_action(get_action_str())
        while True:
            state = self.ddonpach.read_observation()
            # flip_xy(state)
            stabilise(state)

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
