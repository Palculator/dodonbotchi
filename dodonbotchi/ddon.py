import itertools
import logging as log
import math
import numpy as np
import os
import random

from time import sleep

from matplotlib import pyplot as plt
from shapely import geometry

from .mame import Ddonpach, get_action_str
from .util import ensure_directories

TRACK_SPAN = 2
MAX_ENTITY_DISTANCE = 64

VERT = [0, 1, 2]
HORI = [0, 1, 2]
DIRECTIONS = list()

CAUTION_DISTANCE = 64
MIN_DISTANCE = 8

SCREEN_W = 240
SCREEN_H = 320
BOUNDARY = geometry.box(0, 0, SCREEN_W, SCREEN_H)
BOUNDARY_LINE = geometry.LineString(BOUNDARY.exterior)

FIXPOINT = geometry.Point(SCREEN_W / 2, SCREEN_H / 4)


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


class Entity:
    def __init__(self, eid, pos, traj, proj, edge_proj):
        self.eid = eid
        self.pos = pos
        self.traj = traj
        self.proj = proj
        self.edge_proj = edge_proj

    def __str__(self):
        return '({}, {}, {}, {}, {})'.format(self.eid, self.pos, self.traj,
                                             self.proj, self.edge_proj)


def get_trajectory(ent_lst, ent_cur):
    pos_lst = (ent_lst['pos_x'], ent_lst['pos_y'])
    pos_cur = (ent_cur['pos_x'], ent_cur['pos_y'] - TRACK_SPAN + 1)
    traj = (pos_cur[0] - pos_lst[0], pos_cur[1] - pos_lst[1])
    traj = geometry.Point(*traj)
    return traj


def get_projection(pos, traj):
    factor = 3
    proj = geometry.Point(traj.x * factor, traj.y * factor)
    proj = [pos, geometry.Point(pos.x + proj.x, pos.y + proj.y)]
    proj = geometry.LineString(proj)
    return proj


def get_edge_projection(pos, traj):
    factor = 1.0
    while True:
        factor *= 2
        proj = geometry.Point(traj.x * factor, traj.y * factor)
        proj = [pos, geometry.Point(pos.x + proj.x, pos.y + proj.y)]
        proj = geometry.LineString(proj)
        if proj.intersects(BOUNDARY_LINE):
            proj = proj.intersection(BOUNDARY_LINE)
            proj = geometry.LineString([pos, proj])
            return proj


def track_entities(last, current):
    tracked = []
    for idx, ent_lst in enumerate(last):
        if idx >= len(current):
            break

        ent_cur = current[idx]
        if ent_cur['pos_x'] <= 0 or ent_cur['pos_x'] >= 240:
            continue
        if ent_cur['pos_y'] <= 0 or ent_cur['pos_y'] >= 320:
            continue

        if ent_cur['id'] != 0 and ent_cur['id'] == ent_lst['id']:
            pos_lst = geometry.Point(ent_lst['pos_x'], ent_lst['pos_y'])
            pos_cur = geometry.Point(ent_cur['pos_x'], ent_cur['pos_y'])
            if pos_cur.distance(pos_lst) > MAX_ENTITY_DISTANCE:
                continue

            #box = geometry.box(0, 0, ent_cur['siz_x'], ent_cur['siz_y'])

            traj = get_trajectory(ent_lst, ent_cur)
            if traj.x == 0 and traj.y == 0:
                continue

            proj = get_projection(pos_cur, traj)
            edge_proj = get_edge_projection(pos_cur, traj)

            ent = Entity(
                ent_cur['id'],
                pos_cur,
                traj,
                proj,
                edge_proj,
            )

            tracked.append(ent)

    return tracked


def ship_box(ship):
    return geometry.box(ship.x - MIN_DISTANCE,
                        ship.y - MIN_DISTANCE,
                        ship.x + MIN_DISTANCE,
                        ship.y + MIN_DISTANCE)


def plot_line(line, width, colour):
    x_arr, y_arr = line.xy
    plt.plot(x_arr, y_arr, color=colour, linewidth=width,
             solid_capstyle='round')


def plot_entity(ent, colour):
    plot_line(ent.proj, 1.0, colour)


def plot_trajectories(enemies, bullets):
    for enemy in enemies:
        plot_entity(enemy, 'blue')


def plot_objects(objs, mode):
    for obj in objs:
        plt.plot(obj['pos_x'], obj['pos_y'], mode)


def plot_movement(ship, movement, factor):
    direction = DIRECTIONS[movement[0]][movement[1]]
    proj = get_projected_pos(ship, direction, factor)
    line = geometry.LineString([ship, proj])
    plot_line(line, 2.0, 'green')


def plot(ship, observation, bullets):
    # plt.xticks([])
    # plt.yticks([])
    plt.xlim([0, SCREEN_W])
    plt.ylim([0, SCREEN_H])
    plt.axes().set_aspect('equal', 'datalim')

    plot_line(BOUNDARY_LINE, 2.0, 'orange')
    plt.plot(ship.x, ship.y, 'go')
    box = ship_box(ship)
    plot_line(geometry.LineString(box.exterior), 2.0, 'green')

    for bullet in bullets:
        #plt.plot(bullet.pos.x, bullet.pos.y, 'ro')
        plot_line(bullet.proj, 1.0, 'red')

    plot_objects(observation['enemies'], 'bo')
    plot_objects(observation['ownshot'], 'g+')
    plot_objects(observation['powerup'], 'y+')
    plot_objects(observation['bonuses'], 'c+')


def test_direction(ship, bullets, enemies, direction):
    factor = 2.0
    traj = geometry.Point(ship.x + direction[0] * factor,
                          ship.y + direction[1] * factor)

    for bullet in bullets:
        if traj.distance(bullet.proj) < MIN_DISTANCE:
            return None

    return traj


def get_projected_pos(pos, traj, factor):
    proj = geometry.Point(pos.x + traj[0] * factor,
                          pos.y + traj[1] * factor)
    proj = [pos, proj]
    proj = geometry.LineString(proj)
    if False and proj.intersects(BOUNDARY_LINE):
        intersection = proj.intersection(BOUNDARY_LINE)
        proj = [pos, intersection]
        proj = geometry.LineString(proj)
    proj = geometry.Point(*proj.coords[-1])
    return proj


def angle_between(a_x, a_y, b_x, b_y):
    x_diff = b_x - a_x
    y_diff = b_y - a_y
    angle = math.atan2(y_diff, x_diff)
    #angle = (angle + 360) % 360
    return angle


def bullet_distance(ship, vert, hori, bullets, factor, edge=False):
    if not bullets:
        return math.sqrt(240 ** 2 + 320 ** 2)

    direction = DIRECTIONS[vert][hori]
    proj = get_projected_pos(ship, direction, factor)
    distances = []
    for bullet in bullets:
        if bullet.pos.distance(ship) > CAUTION_DISTANCE:
            continue

        beg_dist = proj.distance(geometry.Point(*bullet.proj.coords[-0]))
        end_dist = proj.distance(geometry.Point(*bullet.proj.coords[-1]))
        bullet_direction = np.sign(end_dist - beg_dist)

        if edge:
            distance = bullet.edge_proj.distance(proj)
        else:
            distance = bullet.proj.distance(proj)

        #distance = bullet.pos.distance(proj)
        #distance *= bullet_direction
        distances.append(distance)
    return sum(distances)
    return min(distances)


def enemy_distance(ship, vert, hori, enemies):
    if not enemies:
        return math.sqrt(240 ** 2 + 320 ** 2)

    factor = 4
    direction = DIRECTIONS[vert][hori]
    proj = get_projected_pos(ship, direction, factor)
    distances = []
    for enemy in enemies:
        enemy_pos = geometry.Point(enemy['pos_x'], enemy['pos_y'] -
                                   enemy['siz_y'] - 32)
        if enemy_pos.y > 0:
            distances.append(proj.distance(enemy_pos))
    if distances:
        return min(distances)
    return math.sqrt(240 ** 2 + 320 ** 2)


def object_distance(ship, vert, hori, objects):
    if not objects:
        return math.sqrt(240 ** 2 + 320 ** 2)

    factor = 2
    direction = DIRECTIONS[vert][hori]
    proj = get_projected_pos(ship, direction, factor)
    distances = []
    for obj in objects:
        obj_pos = geometry.Point(obj['pos_x'], obj['pos_y'])

        if obj_pos.y > 0:
            distances.append(proj.distance(obj_pos))
    if distances:
        return min(distances)
    return math.sqrt(240 ** 2 + 320 ** 2)


def is_caution(ship, bullets):
    box = ship_box(ship)
    for bullet in bullets:
        # if bullet.proj.intersects(box):
            # return True
        if bullet.pos.distance(ship) < CAUTION_DISTANCE:
            return True
    return False


def is_powerup(powerup):
    return len(powerup) > 0


def pick_fixed_point(ship):
    movements = []
    for vert, hori in itertools.product(VERT, HORI):
        factor = 2
        direction = DIRECTIONS[vert][hori]
        proj = get_projected_pos(ship, direction, factor)
        point_dist = proj.distance(FIXPOINT)
        movements.append((vert, hori, point_dist))

    movements = list(sorted(movements, key=lambda m: m[2]))
    if movements:
        return movements[0]
    return None


def pick_closest_powerup(ship, powerup):
    movements = []
    for vert, hori in itertools.product(VERT, HORI):
        powerup_dist = object_distance(ship, vert, hori, powerup)
        movements.append((vert, hori, powerup_dist))

    movements = list(sorted(movements, key=lambda m: m[2]))
    if movements:
        return movements[0]
    return None


def pick_closest_enemy(ship, bullets, enemies):
    movements = []
    for vert, hori in itertools.product(VERT, HORI):
        bullets_dist = bullet_distance(ship, vert, hori, bullets, MIN_DISTANCE)
        enemies_dist = enemy_distance(ship, vert, hori, enemies)
        movements.append((vert, hori, bullets_dist, enemies_dist))

    movements = sorted(movements, key=lambda m: m[3])
    movements = [m for m in movements if m[2] > MIN_DISTANCE]
    if movements:
        return movements[0]
    return None


def pick_furthest_bullet(ship, bullets):
    factor = 1
    while factor < 4:
        movements = [(v, h, bullet_distance(ship, v, h, bullets, factor,
                                            edge=True))
                     for v, h in itertools.product(VERT, HORI)]
        movements = list(reversed(sorted(movements, key=lambda m: m[2])))
        factor += 1.0
        for m in movements:
            pass
            #plot_movement(ship, m, MIN_DISTANCE)
        #movements = [m for m in movements if m[2] > MIN_DISTANCE]
        if movements:
            return movements[0]
    return None


def pick_movement(ship, bullets, observation):
    if is_caution(ship, bullets):
        return pick_furthest_bullet(ship, bullets)
    if is_powerup(observation['powerup']):
        return pick_closest_powerup(ship, observation['powerup'])
    return pick_fixed_point(ship)
    # return pick_closest_enemy(ship, bullets, enemies)


def pick_emergency_bomb(ship, bullets):
    for bullet in bullets:
        bullet_pos = geometry.Point(bullet['pos_x'], bullet['pos_y'])
        if ship.distance(bullet_pos) <= 8:
            return 1
    return 0


def pick_action(step, ship, bullets, observation):
    movement = pick_movement(ship, bullets, observation)
    if movement:
        vert = movement[0]
        hori = movement[1]
    else:
        vert, hori = random.choice(VERT), random.choice(HORI)

    bomb = pick_emergency_bomb(ship, observation['bullets'])
    if bomb == 1:
        shot = 1
    else:
        shot = step % 2

    return get_action_str(vert=vert, hori=hori, shot=shot, bomb=bomb)


def ddonpai_step(ddonpach, step, last, current, draw=False):
    ship = geometry.Point(current['ship']['y'], current['ship']['x'])

    bullets = track_entities(last['bullets'], current['bullets'])
    if draw:
        plot(ship, current, bullets)

    action = pick_action(step, ship, bullets, current)
    ddonpach.send_action(action)


def flip_xy(entities):
    for entity in entities:
        entity['pos_x'], entity['pos_y'] = entity['pos_y'], entity['pos_x']
        entity['siz_x'], entity['siz_y'] = entity['siz_y'], entity['siz_x']


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


def play(ddonpai_dir, draw=False, show=False):
    ensure_directories(ddonpai_dir)
    ddonpai_dir = os.path.abspath(ddonpai_dir)

    ddonpach = Ddonpach()
    ddonpach.inp_dir = ddonpai_dir
    ddonpach.snp_dir = ddonpai_dir

    ddonpach.start_mame()

    ddonpai_loop(ddonpach, draw=draw, show=show)
