"""
This module implements classes related to giving access to MAME and DoDonPachi
as an OpenAI-Gym-like environment.
"""
import copy
import json
import logging as log
import math
import os
import random
import socket
import subprocess

from time import sleep

import numpy as np

from jinja2 import Environment, FileSystemLoader
from PIL import Image, ImageDraw
from rl.core import Env, Space

from dodonbotchi.config import CFG as cfg
from dodonbotchi.util import ensure_directories

RECORDING_FILE = 'recording.inp'

IMG_WIDTH = 240
IMG_HEIGHT = 320

OBS_SCALE = 4
OBS_WIDTH = IMG_WIDTH // OBS_SCALE
OBS_HEIGHT = IMG_HEIGHT // OBS_SCALE
OBS_CHANNELS = 'L'

COLOUR_BACKGROUND = '#000000'

COLOUR_ENEMIES = '#AAAAAA'
COLOUR_BONUSES = '#444444'
COLOUR_POWERUP = '#666666'
COLOUR_BULLETS = '#DDDDDD'
COLOUR_OWNSHOT = '#222222'

COLOUR_COMBO = '#888888'

COLOUR_SHIP = '#FFFFFF'

PLUGIN_NAME = 'dodonbotchi_mame'

MAX_COMBO = 0x37
MAX_DISTANCE = 400  # Furthest distance two objects can have in 240x320
COMBO_BAR_HEIGHT = 4

INPUT_SHAPE = (OBS_WIDTH, OBS_HEIGHT)


class DoDonPachiActions(Space):
    """
    This class defines the valid action space for DoDonBotchi. Actions are
    encoded as strings representing button and direction states. The format of
    these strings is:

        VH1

    Where V stands for the vertical axis, H for the horizontal axis, and 1 for
    button 1. An actual action replaces each of these with the corresponding
    input state:

        210

    This example represents positive movement along the V axis, negative
    movement along the H axis, and not pressing 1. The same directions, but
    pressing 1, would be `211`, and so on.

    """

    def __init__(self, seed=None):
        self.directions = 'VH'
        self.buttons = '1'
        self.axis_states = '012'
        self.button_states = '01'

        self.shape = len(self.axis_states) ** len(self.directions)
        self.shape *= len(self.button_states) ** len(self.buttons)
        self.shape = (self.shape,)

        self.rng = random.Random()
        if seed:
            self.rng.seed(seed)

    def from_ordinal(self, ordinal):
        """
        Converts an ordinal integer representing an action to an action encoded
        in the string format outlined above.
        """
        if ordinal >= self.shape[0]:
            raise ValueError('Invalid action ordinal: {}'.format(ordinal))

        action = ''
        base = len(self.axis_states)
        for _ in self.directions:
            idx = ordinal % base
            action += self.axis_states[idx]
            ordinal //= base
        base = len(self.button_states)
        for _ in self.buttons:
            idx = ordinal % base
            action += self.button_states[idx]
            ordinal //= base

        assert len(action) == len(self.directions) + len(self.buttons)
        return action

    def sample(self, seed=None):
        """
        Creates a random DoDonPachi input action.
        """
        if seed:
            self.rng.seed(seed)

        ret = ''
        for _ in self.directions:
            state = self.rng.choice(self.axis_states)
            ret += state
        for _ in self.buttons:
            state = self.rng.choice(self.button_states)
            ret += state
        return ret

    def contains(self, action):
        """
        Tests if the given action string is a member of this action space and
        returns True iff it is.
        """
        if not isinstance(action, str):
            return False
        if len(action) != len(self.buttons) + len(self.directions):
            return False

        for state in action[:len(self.directions)]:
            if state not in self.axis_states:
                return False

        for state in action[len(self.directions):]:
            if state not in self.button_states:
                return False

        return True


def get_point_distance(a_x, a_y, b_x, b_y, root=False):
    """
    Returns the distance between the points a and b, given as x, y coordinate
    pairs. If the root flag is set, the actual distance between them is
    returned, otherwise, the squared distance is returned. This helps save
    processing time when distances are only used for comparisons.
    """
    a = b_x - a_x
    b = b_y - a_y
    dist = a ** 2 + b ** 2
    if root:  # sqrt is an expensive operation so we make it optional
        dist = math.sqrt(dist)
    return dist


def find_closest_object(needle_x, needle_y, haystack):
    """
    Finds the object with the shortest distance to the given x, y coordinates
    within the given list of objects. Objects are expected as dictionaries that
    contain their position in `pos_x` and `pos_y` entries.
    """
    min_dist = MAX_DISTANCE ** 2
    min_obj = None
    for obj in haystack:
        obj_x = obj['pos_x']
        obj_y = obj['pos_y']
        obj_dist = get_point_distance(needle_x, needle_y, obj_x, obj_y)
        if obj_dist < min_dist:
            min_dist = obj_dist
            min_obj = obj
    return min_obj, math.sqrt(min_dist)


def grade_observation(obs):
    """
    Grades the quality of the given observation, returning a value between 0.0
    and 1.0 -- the higher the better.

    The grading criteria are:

        - Shortest distance of an own shot to an enemy
        - Shortest distance of an enemy shot to our ship
        - Combo timer

    A perfect grade is rewarded when the shortest distance from one of our
    shots to an enemy is 1, the shortest distance of an enemy shot to our ship
    is 400, and the combo timer is full.

    Special cases are when there are simply no enemies or bullets on screen; in
    those cases, the function assigns a perfect score to those components of
    the grade.

    The resulting grade is returned as a number.
    """
    ship = obs['ship']
    ship_x = ship['x']
    ship_y = ship['y']

    combo = obs['combo']

    enemies = obs['enemies']
    bullets = obs['bullets']
    ownshot = obs['ownshot']

    bullet_reward = 1
    enemy_reward = 1
    combo_reward = combo / MAX_COMBO

    log.info('Current combo timer is: %s', combo)
    log.info('Determined combo reward to be: %s', combo_reward)

    bullet, bullet_dist = find_closest_object(ship_x, ship_y, bullets)
    if bullet:
        log.debug('Closest bullet at %s: %s: %s, %s',
                  bullet_dist,
                  bullet['id'],
                  bullet['pos_x'],
                  bullet['pos_y'])
        bullet_reward = bullet_dist / MAX_DISTANCE
    else:
        log.debug('No bullet on screen.')
    log.info('Determined bullet reward to be: %s', bullet_reward)

    enemy, enemy_dist = None, MAX_DISTANCE
    for own in ownshot:
        own_x = own['pos_x']
        own_y = own['pos_y']
        cur_enemy, cur_dist = find_closest_object(own_x, own_y, enemies)
        if cur_dist < enemy_dist:
            enemy = cur_enemy
            enemy_dist = cur_dist
    if enemy:
        if enemy_dist == 0:
            enemy_dist = 1  # Avoid division by 0
        log.debug('Closest enemy at %s: %s: %s, %s',
                  enemy_dist,
                  enemy['id'],
                  enemy['pos_x'],
                  enemy['pos_y'])
        enemy_reward = 1.0 / enemy_dist
    else:
        log.debug('No enemy/ownshot on screen.')
    log.info('Determined enemy reward to be: %s', enemy_reward)

    reward = combo_reward + bullet_reward + enemy_reward
    reward /= 3
    log.info('Determined observation reward to be: %s', reward)

    return reward


def draw_objects(draw, objects, colour, scale=2):
    """
    Draws the given list of objects as boxes to the given ImageDraw object
    using the given colour. Objects are expected to be dictionaries containing
    their center position as `pos_x` and `pos_y` entries, and the object's size
    as `siz_x` and `siz_y` entries.
    """
    for obj in objects:
        off_x = obj['siz_x'] / scale
        off_y = obj['siz_y'] / scale

        min_x = OBS_HEIGHT - obj['pos_x'] - off_x
        min_y = obj['pos_y'] - off_y
        max_x = OBS_HEIGHT - obj['pos_x'] + off_x
        max_y = obj['pos_y'] + off_y

        draw.rectangle((min_y, min_x, max_y, max_x), fill=colour)


def scale_objects(objects):
    """
    Scales the given list of objects down in-place, dividing their positions
    and sizes by `OBS_SCALE`.
    """
    for obj in objects:
        obj['pos_x'] = obj['pos_x'] // OBS_SCALE
        obj['pos_y'] = obj['pos_y'] // OBS_SCALE
        obj['siz_x'] = obj['siz_x'] // OBS_SCALE
        obj['siz_y'] = obj['siz_y'] // OBS_SCALE


def observation_to_image(obs):
    """
    Renders the given observation dictionary into an image that represents the
    game state described in the observation. The resulting image is returned as
    a pillow Image object.
    """
    obs = copy.deepcopy(obs)

    img = Image.new(OBS_CHANNELS, (OBS_WIDTH, OBS_HEIGHT), COLOUR_BACKGROUND)
    draw = ImageDraw.Draw(img)

    enemies = obs['enemies']
    bullets = obs['bullets']
    ownshot = obs['ownshot']
    powerup = obs['powerup']
    bonuses = obs['bonuses']

    scale_objects(enemies)
    scale_objects(bullets)
    scale_objects(ownshot)
    scale_objects(powerup)
    scale_objects(bonuses)

    ship_obj = {
        'pos_x': obs['ship']['x'] // OBS_SCALE,
        'pos_y': obs['ship']['y'] // OBS_SCALE,
        'siz_x': 16 // OBS_SCALE,
        'siz_y': 16 // OBS_SCALE
    }

    draw_objects(draw, enemies, COLOUR_ENEMIES)
    draw_objects(draw, ownshot, COLOUR_OWNSHOT, 4)
    draw_objects(draw, bonuses, COLOUR_BONUSES, 4)
    draw_objects(draw, powerup, COLOUR_POWERUP)
    draw_objects(draw, bullets, COLOUR_BULLETS, 4)

    draw_objects(draw, [ship_obj], COLOUR_SHIP)

    combo_width = int((obs['combo'] / MAX_COMBO) * OBS_WIDTH)
    if combo_width:
        rect = (0, OBS_HEIGHT - COMBO_BAR_HEIGHT, combo_width, OBS_HEIGHT)
        draw.rectangle(rect, fill=COLOUR_COMBO)

    del draw

    return img


class DoDonPachiObservations(Space):
    """
    Defines the observation space for DoDonPachi. Observations are grayscale
    images of size OBS_WIDTH x OBS_HEIGHT.
    """

    def __init__(self, seed=None):
        self.rng = random.Random()
        if seed:
            self.rng.seed(seed)

        self.shape = INPUT_SHAPE


def get_plugin_path():
    """
    Gets the target directory to save the dodonbotchi plugin to relevant to the
    MAME home directory specified in the global config.
    """
    plugin_path = cfg.mame_path
    plugin_path = os.path.join(plugin_path, 'plugins')
    plugin_path = os.path.join(plugin_path, PLUGIN_NAME)
    return plugin_path


def write_plugin(**options):
    """
    Renders the templates for the code of the dodonbotchi plugin and writes it
    to the appropriate plugin directory.
    """
    plugin_path = get_plugin_path()
    ensure_directories(plugin_path)

    templates_path = os.path.join('dodonbotchi/plugin', PLUGIN_NAME)
    templates_env = Environment(loader=FileSystemLoader(templates_path))

    for template_name in os.listdir(templates_path):
        template_path = os.path.join(templates_path, template_name)
        if os.path.isfile(template_path):
            template = templates_env.get_template(template_name)
            rendered = template.render(plugin_name=PLUGIN_NAME, **options)

            template_target = os.path.join(plugin_path, template_name)
            with open(template_target, 'w') as out_file:
                out_file.write(rendered)


def generate_base_call():
    """
    Generates a list of parameters to start MAME with DoDonPachi which contain
    command line options corresponding to what's configured in the global
    configuration. The list gets returned and can be customised by appending
    further options.
    """
    call = ['mame', 'ddonpach', '-skip_gameinfo', '-pause_brightness', '1']

    if cfg.windowed:
        call.append('-window')

    if cfg.nothrottle:
        call.append('-nothrottle')

    if cfg.novideo:
        call.append('-video')
        call.append('none')

    if cfg.noaudio:
        call.append('-sound')
        call.append('none')

    if cfg.save_state:
        call.append('-state')
        call.append(cfg.save_state)

    return call


def render_avi(inp_file, avi_file, inp_dir=None, snp_dir=None):
    """
    Plays back input file at the given path recording it as a video to the
    given .avi file path. Optionally, recording and capture directories can be
    overridden using `inp_dir` and `snp_dir`.
    """
    write_plugin(mode='record', **cfg)

    call = generate_base_call()

    call.append('-plugin')
    call.append(PLUGIN_NAME)

    if inp_dir:
        call.append('-input_directory')
        call.append(inp_dir)

    call.append('-playback')
    call.append(inp_file)
    call.append('-exit_after_playback')

    if snp_dir:
        call.append('-snapshot_directory')
        call.append(snp_dir)

    call.append('-aviwrite')
    call.append(avi_file)

    return subprocess.call(call)


class DoDonPachiEnv(Env):
    """
    Implements an OpenAI-Gym-like envirionment that starts and interfaces with
    MAME running DoDonPachi. The respective action and observation spaces are
    defined in the DoDonPachiActions and DoDonPachiObservations classes. What
    this class does is offer a way to perform actions within MAME-emulated
    DoDonPachi through a socket and retrieve observations from there. Lives and
    scores in those observations are tracked to compute action rewards.
    """

    def __init__(self, seed=None):
        self.reward_range = (-1, 1)
        self.action_space = DoDonPachiActions(seed)
        self.observation_space = DoDonPachiObservations(seed)

        self.obs_count = 0

        self.inp_dir = None
        self.snp_dir = None

        self.process = None
        self.server = None
        self.client = None
        self.sfile = None

        self.current_frame = 0
        self.current_lives = 0
        self.current_score = 0
        self.current_bombs = 0
        self.current_combo = 0
        self.current_hit = 0
        self.reward_sum = 0
        self.current_observation = None
        self.current_grade = 1.0
        self.current_reward = 0
        self.max_hit = -1

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((cfg.host, cfg.port))
        self.server.listen()
        log.info('Started socket server on %s:%s', cfg.host, cfg.port)

        write_plugin(mode='bot', **cfg)

    def configure(self, *args, **options):
        pass

    def seed(self, seed=None):
        """
        Stub because this environment has no random properties to control with
        a seed.
        """
        return [seed]

    def send_message(self, message):
        """
        Sends a message to the client, terminated by a newline.
        """
        self.sfile.write('{}\n'.format(message))
        self.sfile.flush()

    def send_command(self, command, **options):
        """
        Sends a command to the client in the form of a json object containing
        at least a `command` field and additional fields given in the
        **options.
        """
        message = {'command': command}
        for key, val in options.items():
            message[key] = val
        message = json.dumps(message)
        self.send_message(message)

    def send_action(self, action):
        """
        Sends an action to perform to the client. The given action must be a
        member of the DoDonPachiActions space.
        """
        self.send_command('action', inputs=action)

    def read_message(self):
        """
        Reads a message from the client, expecting it to be in one line and a
        json object. The method returns the message parsed as a dictionary.
        """
        line = self.sfile.readline()
        return json.loads(line)

    def dump_observation_frame(self, observation):
        """
        Retrieves the most recently saved snapshot of DoDonPachi, appends the
        artificial observation frame to its right and overwrites that file.
        """
        dump = Image.new('RGB', (IMG_WIDTH * 2, IMG_HEIGHT), COLOUR_BACKGROUND)

        snapshots = os.path.join(self.snp_dir, 'ddonpach')

        if os.path.exists(snapshots):
            snapshot_file = sorted(os.listdir(snapshots))[-1]
            snapshot_path = os.path.join(snapshots, snapshot_file)
            snapshot = Image.open(snapshot_path)
            dump.paste(snapshot, (0, 0))

            observation = observation.resize((IMG_WIDTH, IMG_HEIGHT))
            dump.paste(observation, (IMG_WIDTH, 0))

            dump.save(snapshot_path, 'PNG')

    def read_observation(self):
        """
        Reads an observation from the client and renders an artificial frame
        containing the objects described in the observation. The image is
        returned as a numpy array alongside the observation dictionary
        originally sent by the client.
        """
        message = self.read_message()
        observation_dic = message['observation']
        self.current_frame = observation_dic['frame']

        img = observation_to_image(observation_dic)

        if cfg.dump_frames:
            self.dump_observation_frame(img)

        arr = np.array(img)
        arr = np.reshape(arr, self.observation_space.shape)

        self.obs_count += 1

        assert arr.shape == self.observation_space.shape

        return arr, observation_dic

    def start_mame(self):
        """
        Boots up MAME with the globally configured options and additionally
        setting it to record inputs this environment's respective folders for
        those.
        """
        call = generate_base_call()
        call.append('-plugin')
        call.append(PLUGIN_NAME)

        abs_inp_dir = os.path.abspath(self.inp_dir)
        call.append('-input_directory')
        call.append(abs_inp_dir)
        call.append('-record')
        call.append(RECORDING_FILE)

        abs_snp_dir = os.path.abspath(self.snp_dir)
        call.append('-snapshot_directory')
        call.append(abs_snp_dir)

        self.process = subprocess.Popen(call)
        log.info('Started MAME with dodonbotchi ipc & dodonpachi.')
        log.info('Waiting for MAME to connect...')

        self.client, addr = self.server.accept()
        self.sfile = self.client.makefile(mode='rw')
        log.info('Accepted client from: %s', addr)

    def stop_mame(self):
        """
        Tries to gracefully terminate MAME by sending the client the kill
        command, but killing the process manually if the client does not
        terminate on its own.
        """
        assert self.process

        if self.client and self.sfile:
            self.send_command('kill')

        sleep(1)

        for _ in range(10):
            if not self.process:
                break

            log.debug('Waiting for MAME to die...')
            try:
                os.kill(self.process.pid, 0)
            except OSError:
                sleep(0.5)
                continue

            self.process = None

        if self.process:
            os.kill(self.process.pid, 9)
            self.process = None

        self.client = None
        self.sfile = None

    def step(self, action):
        """
        Performs the given action in this environment. The action can either be
        an integer representing an ordinal value in our action space, or a
        string formatted as defined in DoDonPachiActions. The method returns a
        tuple of (observation, reward, done, aux) representing the observation
        after the action was performed, the reward gained from it, whether the
        simulation is done, and auxiliary data to give additional and optional
        info.
        """
        if isinstance(action, np.integer):
            action = self.action_space.from_ordinal(action)
        if not self.action_space.contains(action):
            raise ValueError('Action not in action space: {}'.format(action))

        log.debug('Performing DoDonBotchi action: %s', action)
        self.send_action(action)
        log.debug('Action sent. Waiting for observation...')
        observation, observation_dic = self.read_observation()

        grade = grade_observation(observation_dic)
        reward = self.current_grade - grade

        self.reward_sum += reward

        lives = observation_dic['lives']
        score = observation_dic['score']
        combo = observation_dic['combo']
        bombs = observation_dic['bombs']
        hit = observation_dic['hit']

        score_difference = score - self.current_score
        score_difference += 1  # Ensure we don't 0 out the reward

        log.info('Got step reward: %s', reward)

        if lives < self.current_lives:
            reward = -1

        done = lives == 2

        self.current_lives = lives
        self.current_score = score
        self.current_combo = combo
        self.current_bombs = bombs
        self.current_hit = hit
        self.current_observation = observation_dic
        self.current_reward = reward
        self.current_grade = grade

        if self.current_hit > self.max_hit:
            self.max_hit = self.current_hit

        return observation, reward, done, {}

    def reset_stats(self):
        self.current_frame = 0
        self.current_lives = 0
        self.current_score = 0
        self.current_bombs = 0
        self.current_combo = 0
        self.current_grade = 1
        self.current_reward = 0
        self.current_hit = 0
        self.reward_sum = 0
        self.max_hit = -1

    def reset(self):
        """
        Resets MAME and DoDonPachi to start from scratch. The initial
        observation immediately after starting the game is returned.
        """
        self.current_observation = None

        ensure_directories(self.inp_dir, self.snp_dir)

        if self.process:
            self.stop_mame()

        self.start_mame()

        observation, observation_dic = self.read_observation()

        self.current_lives = observation_dic['lives']
        self.current_score = observation_dic['score']
        self.current_combo = observation_dic['combo']
        self.current_bombs = observation_dic['bombs']
        self.current_hit = observation_dic['hit']

        return observation

    def render(self, mode='human', close=False):
        """
        Meaningless function in our context, since rendering is turned off/on
        in the global config that gets passed to MAME. Only implemented as part
        of the Env interface.
        """
        # Kind of meaningless in our setup, so this method is empty.
        pass

    def close(self):
        """
        Kills MAME and closes the server socket.
        """
        if self.process:
            self.stop_mame()

        if self.server:
            self.server.close()

        self.process = None
        self.server = None
        self.client = None
        self.sfile = None
