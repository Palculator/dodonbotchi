"""
This module contains DoDonBotchi's brain and neuroevolutional code. It handles
learning, evolution, and controlling DoDonPachi in realtime.
"""
import json
import logging as log
import os
import os.path

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback
from rl.memory import EpisodeParameterMemory, SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy
from rl.policy import BoltzmannGumbelQPolicy

from dodonbotchi.config import CFG as cfg
from dodonbotchi.mame import DoDonPachiEnv
from dodonbotchi.util import get_now_string, generate_now_serial_number
from dodonbotchi.util import ensure_directories


REC_DIR = 'recordings'
CAP_DIR = 'captures'

BRAIN_FILE = 'brain.h5f'
LEADERBOARD_FILE = 'leaderboard.json'


def create_model(actions, observations):
    """
    Creates the neural network model using the given observation and action
    dimensions and returns it alongside the number of nodes in the network.
    """
    num = 256 + 512 + 512 + actions + observations

    model = Sequential()
    model.add(Flatten(input_shape=(5, observations)))
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

    return model, num


def create_memory():
    """
    Creates a memory instance to be used by a learning agent and returns it
    alongside a memory model type to identify it.
    """
    memory = SequentialMemory(limit=100000, window_length=5)
    return memory, 'S'


def create_policy():
    """
    Creates a policy instance to be used by a learning agent and returns it
    alongside a policy model type to identify it.
    """
    policy = EpsGreedyQPolicy()
    policy = LinearAnnealedPolicy(policy, attr='eps', value_max=1.0,
                                  value_min=0.1, value_test=0.05,
                                  nb_steps=10000000)
    return policy, 'LAeg'


def create_agent(actions, observations):
    """
    Creates the reinforcement learning agent used by EXY, with an underlying
    neural network using the given observation and action dimensions.

    The agent is returned along with a serial number identifying it.
    """
    model, nodes = create_model(actions, observations)
    memory, mem_type = create_memory()
    policy, pol_type = create_policy()

    agent = DQNAgent(model=model, memory=memory, policy=policy,
                     nb_actions=actions, nb_steps_warmup=500)
    agent.compile(Adam(lr=.00025), metrics=['mae'])

    mem_map = {'S': '13', 'E': '9K'}
    mem_type = mem_map[mem_type]

    serial = '{}{}-{}-{:03}'.format('DQN', mem_type, pol_type, nodes % 1000)
    return agent, serial


def get_recording_dir(exy_dir):
    return os.path.join(exy_dir, REC_DIR)


def get_capture_dir(exy_dir):
    return os.path.join(exy_dir, CAP_DIR)


def get_brain_file(exy_dir):
    return os.path.join(exy_dir, BRAIN_FILE)


def get_leaderboard_file(exy_dir):
    return os.path.join(exy_dir, LEADERBOARD_FILE)


class EXY(Callback):
    """
    The EXY class is used to gradually train a neural net to perform well in
    DoDonPachi. It's initialised with a working directory it will store
    training data, including a brain file containing the current weights of the
    neural network, to. This folder includes recordings of the AI's run,
    leaderboards, and regular snapshots.

    If the directory already contains a brain file named as defined in
    `BRAIN_FILE`, the current state of the brain will be loaded for further
    training.
    """

    def __init__(self, exy_dir):
        self.exy_dir = exy_dir

        self.brain_file = get_brain_file(self.exy_dir)

        self.agent = None
        self.serial = None

        self.inp_dir = None
        self.snp_dir = None

        self.env = None
        self.leaderboard = []
        self.leaderboard_file = get_leaderboard_file(self.exy_dir)
        if os.path.exists(self.leaderboard_file):
            with open(self.leaderboard_file, 'r') as in_file:
                leaderboard_text = in_file.read()
                leaderboard_json = json.loads(leaderboard_text)
                self.leaderboard = leaderboard_json['leaderboard']

    def setup(self):
        """
        Setups up the agent instance and output directories based on the given
        environment.
        """
        actions = self.env.action_space.count
        observations = self.env.observation_space.dimension

        self.agent, self.serial = create_agent(actions, observations)

        self.inp_dir = get_recording_dir(self.exy_dir)
        self.snp_dir = get_capture_dir(self.exy_dir)

        ensure_directories(self.exy_dir, self.inp_dir, self.snp_dir)

        if os.path.exists(self.brain_file):
            self.agent.load_weights(self.brain_file)

    def on_episode_end(self, episode, logs={}):
        entry = {'inp': self.env.recording, 'score': self.env.current_score}
        self.leaderboard.append(entry)
        self.leaderboard = sorted(self.leaderboard, key=lambda e: e['score'])
        self.leaderboard.reverse()
        if len(self.leaderboard) > 32:
            self.leaderboard = self.leaderboard[:32]

        log.info('!!! Current run ended with score: {}'.format(
            self.env.current_score))

        with open(self.leaderboard_file, 'w') as out_file:
            leaderboard_dict = {'leaderboard': self.leaderboard}
            leaderboard_json = json.dumps(leaderboard_dict, indent=4)
            out_file.write(leaderboard_json)

    def train(self):
        """
        Creates a DoDonPachi environment and a reinforcement learning agent for
        it and then trains it on that environment. Training can be interrupted
        and after training, either manually stopped or regularly terminated,
        current weights will be saved to EXY's brain file.
        """
        self.env = DoDonPachiEnv()
        self.setup()
        try:
            self.env.configure(inp_dir=self.inp_dir, snp_dir=self.snp_dir)
            self.env.reset()
            self.agent.fit(self.env, callbacks=[self],
                           nb_steps=10000000000, verbose=2)
        finally:
            self.env.close()
            self.agent.save_weights(self.brain_file, overwrite=True)
