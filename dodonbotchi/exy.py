"""
This module contains DoDonBotchi's brain and neuroevolutional code. It handles
learning, evolution, and controlling DoDonPachi in realtime.
"""
import logging as log
import os
import os.path

from dodonbotchi.config import CFG as cfg
from dodonbotchi.mame import DoDonPachiEnv
from dodonbotchi.util import get_now_string, generate_now_serial_number
from dodonbotchi.util import ensure_directories

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten
from tensorflow.python.keras.optimizers import Adam


REC_DIR = 'recordings'
CAP_DIR = 'captures'

BRAIN_FILE_FMT = 'brain_{}.h5f'
PROPS_FILE_FMT = 'props_{}.json'


def create_model(actions, observations):
    model = Sequential()
    model.add(Flatten(input_shape=(1, observations)))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('softmax'))

    log.info('Created neural network model: ')
    log.info(model.summary())

    return model


def create_agent(actions, observations):
    model = create_model(actions, observations)
    memory = EpisodeParameterMemory(limit=1000, window_length=1)
    cem = CEMAgent(model=model, nb_actions=actions, memory=memory)
    cem.compile()
    return cem


class EXY:
    def __init__(self, output):
        self.output = output
        self.serial = generate_now_serial_number()

        self.exy_dir = get_now_string()
        self.exy_dir = '{} - {}'.format(self.exy_dir, self.serial)
        self.exy_dir = os.path.join(output, self.exy_dir)

        self.inp_dir = os.path.join(self.exy_dir, REC_DIR)
        self.snp_dir = os.path.join(self.exy_dir, CAP_DIR)

        ensure_directories(self.exy_dir, self.inp_dir, self.snp_dir)

    def train(self, seed=None):
        env = DoDonPachiEnv(seed)
        try:
            env.configure(inp_dir=self.inp_dir, snp_dir=self.snp_dir)
            env.reset()

            actions = env.action_space.count
            observations = env.observation_space.dimension
            agent = create_agent(actions, observations)

            agent.fit(env, nb_steps=10000000000, verbose=2)
        finally:
            env.close()
