"""
This module specifies available reinforcement learning agents for DoDonBotchi.
"""
import copy
import logging as log
import math
import os

from time import sleep

import numpy as np

from PIL import Image, ImageDraw
from rl.core import Processor

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Permute, Convolution2D, TimeDistributed, LSTM
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback
from rl.memory import EpisodeParameterMemory, SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy
from rl.policy import BoltzmannGumbelQPolicy

from dodonbotchi import mame
from dodonbotchi.config import CFG as cfg

IMG_WIDTH = 320  # Technically it's 240, but Conv2D works better with square frames
IMG_HEIGHT = 320

OBS_SCALE = 4
OBS_WIDTH = IMG_WIDTH // OBS_SCALE
OBS_HEIGHT = IMG_HEIGHT // OBS_SCALE
OBS_CHANNELS = 'L'

COLOUR_BACKGROUND = '#000000'

COLOUR_ENEMIES = '#AAAAAA'
COLOUR_BONUSES = '#222222'
COLOUR_POWERUP = '#666666'
COLOUR_BULLETS = '#444444'
COLOUR_OWNSHOT = '#CCCCCC'

COLOUR_COMBO = '#888888'

COLOUR_SHIP = '#EEEEEE'

COMBO_BAR_HEIGHT = 4

IMG_INPUT_SHAPE = (OBS_WIDTH, OBS_HEIGHT)

IMG_MEMORY_WINDOW = 4

THREAT_RING_RADIUS = 45
THREAT_RING_MAX_DIST = math.sqrt(320 * 320 + 240 * 240)
THREAT_RING_PEAK = 16
THREAT_RING_MEMORY = 2


def dump_observation_frame(snp_dir, observation):
    """
    Retrieves the most recently saved snapshot of DoDonPachi, appends the
    artificial observation frame to its right and overwrites that file.
    """
    dump = Image.new('RGB', (IMG_WIDTH * 2, IMG_HEIGHT), COLOUR_BACKGROUND)

    snapshots = os.path.join(snp_dir, 'ddonpach')

    if os.path.exists(snapshots):
        snapshot_file = sorted(os.listdir(snapshots))[-1]
        snapshot_path = os.path.join(snapshots, snapshot_file)
        snapshot = Image.open(snapshot_path)
        dump.paste(snapshot, (0, 0))

        observation = observation.resize((IMG_WIDTH, IMG_HEIGHT))
        dump.paste(observation, (IMG_WIDTH, 0))

        dump.save(snapshot_path, 'PNG')


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
        obj['siz_x'] = obj['siz_x'] // (OBS_SCALE * 2)
        obj['siz_y'] = obj['siz_y'] // (OBS_SCALE * 2)


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
        'siz_x': 16 // (OBS_SCALE * 2),
        'siz_y': 16 // (OBS_SCALE * 2),
    }

    draw_objects(draw, enemies, COLOUR_ENEMIES)
    draw_objects(draw, ownshot, COLOUR_OWNSHOT)
    draw_objects(draw, [ship_obj], COLOUR_SHIP)
    draw_objects(draw, bonuses, COLOUR_BONUSES)
    draw_objects(draw, powerup, COLOUR_POWERUP)
    draw_objects(draw, bullets, COLOUR_BULLETS)

    combo_width = int((obs['combo'] / mame.MAX_COMBO) * OBS_WIDTH)
    if combo_width:
        rect = (0, OBS_HEIGHT - COMBO_BAR_HEIGHT, combo_width, OBS_HEIGHT)
        draw.rectangle(rect, fill=COLOUR_COMBO)

    del draw

    return img


class FrameProcessor(Processor):
    def __init__(self, exy, env):
        self.exy = exy
        self.env = env

    def process_observation(self, observation):
        img = observation_to_image(observation)

        if cfg.dump_frames:
            dump_observation_frame(self.exy.episode_dir, img)

        arr = np.array(img)
        arr = np.reshape(arr, IMG_INPUT_SHAPE)

        return arr


def create_conv_agent(exy, env, actions):
    input_shape = (IMG_MEMORY_WINDOW,) + IMG_INPUT_SHAPE

    model = Sequential()
    print(input_shape)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten(activation='relu'))
    model.add(Dense(512))
    model.add(Dense(actions))

    log.info('Created convolutional neural network model: ')
    log.info(model.summary())

    memory = SequentialMemory(limit=100000, window_length=IMG_MEMORY_WINDOW)
    log.debug('Created reinforcement learning memory.')

    policy = EpsGreedyQPolicy()
    policy = LinearAnnealedPolicy(policy, attr='eps', value_max=0.9,
                                  value_min=0.1, value_test=0.05,
                                  nb_steps=1000000)
    log.debug('Created reinforcement learning policy.')

    processor = FrameProcessor(exy, env)

    agent = DQNAgent(model=model, memory=memory, policy=policy,
                     nb_actions=actions, nb_steps_warmup=exy.warmup,
                     processor=processor)
    agent.compile(Adam(), metrics=['mae'])
    log.debug('Created reinforcement learning agent.')

    return agent


def angle_between(a_x, a_y, b_x, b_y):
    x_diff = b_x - a_x
    y_diff = b_y - a_y
    angle = math.degrees(math.atan2(y_diff, x_diff)) + 180
    angle = (angle + 360) % 360
    return angle


def distance_between(a_x, a_y, b_x, b_y):
    x_diff = b_x - a_x
    y_diff = b_y - a_y
    distance = x_diff * x_diff + y_diff * y_diff
    distance = math.sqrt(distance)
    return distance


def create_threat_ring(ship_x, ship_y, entities):
    ring = [0.0, ] * (360 // THREAT_RING_RADIUS)
    for entity in entities:
        pos_x = entity['pos_x']
        pos_y = entity['pos_y']

        angle = angle_between(pos_x, pos_y, ship_x, ship_y)
        distance = distance_between(pos_x, pos_y, ship_x, ship_y)

        angle = math.floor(angle / THREAT_RING_RADIUS)
        distance = math.floor(THREAT_RING_MAX_DIST / (distance + 0.01))

        ring[angle] += distance

    return ring


class ThreatRingProcessor(Processor):
    def __init__(self, env):
        self.env = env

    def process_observation(self, observation):
        ship_x = observation['ship']['x']
        ship_y = observation['ship']['y']

        enemies = observation['enemies']
        bullets = observation['bullets']

        enemies_ring = create_threat_ring(ship_x, ship_y, enemies)
        bullets_ring = create_threat_ring(ship_x, ship_y, bullets)

        if cfg.render_ring == "true":
            self.env.send_command('threat_ring',
                                  ship_x=ship_x,
                                  ship_y=ship_y,
                                  enemies=enemies_ring,
                                  bullets=bullets_ring)
            self.env.read_message()

        ret = (ship_x, ship_y, *enemies_ring, *bullets_ring)

        return ret


def create_threat_ring_agent(exy, env, actions):
    threat_ring_segs = 360 // THREAT_RING_RADIUS
    input_shape = (threat_ring_segs * 2 + 2,)  # Two rings and ship x, y
    input_shape = (THREAT_RING_MEMORY, *input_shape)  # Memory

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('relu'))

    log.info('Created Threat Ring neural net model.')
    log.info(model.summary())

    memory = SequentialMemory(limit=10000000, window_length=THREAT_RING_MEMORY)
    log.debug('Created reinforcement learning memory.')

    policy = EpsGreedyQPolicy()
    policy = LinearAnnealedPolicy(policy, attr='eps', value_max=0.9,
                                  value_min=0.1, value_test=0.05,
                                  nb_steps=1000000)
    log.debug('Created reinforcement learning policy.')

    processor = ThreatRingProcessor(env)

    agent = DQNAgent(model=model, memory=memory, policy=policy,
                     nb_actions=actions, nb_steps_warmup=exy.warmup,
                     processor=processor)
    agent.compile(Adam(), metrics=['mae'])
    log.debug('Created reinforcement learning agent.')

    return agent


def create_agent(agent_name, exy, env, actions):
    if agent_name == 'conv':
        return create_conv_agent(exy, env, actions)

    if agent_name == 'threat_ring':
        return create_threat_ring_agent(exy, env, actions)

    return None
