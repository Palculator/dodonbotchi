"""
This module contains DoDonBotchi's brain and neuroevolutional code. It handles
learning, evolution, and controlling DoDonPachi in realtime.
"""
import json
import logging as log
import os
import os.path

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Permute, Convolution2D
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


EPS_DIR = 'episodes'

BRAIN_FILE = 'brain.h5f'
PROPS_FILE = 'exy.json'
STATS_FILE = 'stats.csv'
STOP_FILE = 'stop.pls'

SNAP_DIR = 'snap'

MEMORY_WINDOW = 4
STEPS = 10000000


def get_snap_dir(ep_dir):
    return os.path.join(ep_dir, SNAP_DIR)


def get_stats_file(ep_dir):
    return os.path.join(ep_dir, STATS_FILE)


def generate_episode_serial(ep_num):
    serial = generate_now_serial_number()
    serial = '{:08} - {}'.format(ep_num, serial)
    return serial


def create_model(actions, input_shape):
    """
    Creates the neural network model outputting any of the given amount of
    actions and observing inputs of the given shape.

    This function returns the neural net as an uncompiled keras model.
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('linear'))

    log.info('Created neural network model: ')
    log.info(model.summary())

    return model


def create_memory():
    """
    Creates a memory instance to be used by a learning agent and returns it.
    """
    memory = SequentialMemory(limit=100000, window_length=MEMORY_WINDOW)
    log.debug('Created reinforcement learning memory.')
    return memory


def create_policy():
    """
    Creates a policy instance to be used by a learning agent and returns it.
    """
    policy = EpsGreedyQPolicy()
    policy = LinearAnnealedPolicy(policy, attr='eps', value_max=1.0,
                                  value_min=0.1, value_test=0.05,
                                  nb_steps=STEPS)
    log.debug('Created reinforcement learning policy.')
    return policy


def create_agent(actions, input_shape):
    """
    Creates the reinforcement learning agent used by EXY, with an underlying
    neural network using the given observation and action dimensions.
    """
    input_shape = (MEMORY_WINDOW,) + input_shape
    model = create_model(actions, input_shape)
    memory = create_memory()
    policy = create_policy()

    agent = DQNAgent(model=model, memory=memory, policy=policy,
                     nb_actions=actions, nb_steps_warmup=5000)
    agent.compile(Adam(lr=.00025), metrics=['mae'])
    log.debug('Created reinforcement learning agent.')
    return agent


class EXY(Callback):
    """
    The EXY class is used to gradually train a neural net to perform well in
    DoDonPachi. It's initialised with a working directory it will store
    training data, including a brain file containing the current weights of the
    neural network, to. This folder includes recordings of the AI's run,
    leaderboards, and regular snapshots.

    If the directory already contains a brain file named as defined in
    `BRAIN_FILE`, the current state of the brain will be loaded for further
    training. Similarly, a properties file `PROPS_FILE` will be read to resume
    EXY's state from where it was left off.
    """

    def __init__(self, exy_dir):
        self.exy_dir = exy_dir

        self.env = None

        self.episode_num = 0
        self.episode_ser = None
        self.leaderboard = []

        self.current_stats = None

    def get_brain_file(self):
        """Returns the path to the brain file of this EXY instance."""
        return os.path.join(self.exy_dir, BRAIN_FILE)

    def get_props_file(self):
        """Returns the path to the properties file of this EXY instance."""
        return os.path.join(self.exy_dir, PROPS_FILE)

    def get_episodes_dir(self):
        """Returns the path to the episodes directory of this EXY instance."""
        return os.path.join(self.exy_dir, EPS_DIR)

    def get_episode_dir(self, ep_ser):
        """
        Returns the path to the episode directory of the given serial number in
        this EXY instance.
        """
        episodes = self.get_episodes_dir()
        path = os.path.join(episodes, ep_ser)
        return path

    def get_stop_file(self):
        """
        Returns the path to the file created when EXY is supposed to stop.
        """
        return os.path.join(self.exy_dir, STOP_FILE)

    def load_properties(self):
        """
        Loads previously saved EXY properties from the properties file in this
        EXY instance's working directory. If no such file exists, nothing
        happens.
        """
        props_file = self.get_props_file()
        if not os.path.exists(props_file):
            return

        with open(props_file, 'r') as in_file:
            props_str = in_file.read()
            props_dic = json.loads(props_str)

            self.episode_num = props_dic['episode_num']
            self.leaderboard = props_dic['leaderboard']
            log.debug('Loaded EXY properties from: %s', props_file)

    def save_properties(self):
        """
        Saves this EXY instance's properties to a properties file in this
        instance's working directory. An existing file would be overwritten.
        """
        props_file = self.get_props_file()
        with open(props_file, 'w') as out_file:
            props_dic = dict()
            props_dic['episode_num'] = self.episode_num
            props_dic['leaderboard'] = self.leaderboard
            props_str = json.dumps(props_dic, indent=4)

            out_file.write(props_str)
            log.debug('Saved EXY properties to: %s', props_file)

    def load_brain(self, agent):
        """
        Loads weights from a previously saved brain file into the given agent.
        The function looks for the file in this EXY instance's working
        directory. If no file is found, nothing happens.
        """
        brain_file = self.get_brain_file()
        if not os.path.exists(brain_file):
            return

        agent.load_weights(brain_file)
        log.debug('Loaded brain from: %s', brain_file)

    def save_brain(self, agent):
        """
        Saves the given agent's weights to a brain file in this EXY instance's
        working directory.
        """
        brain_file = self.get_brain_file()
        agent.save_weights(brain_file, overwrite=True)
        log.debug('Saved brain to: %s', brain_file)

    def setup_next_episode(self):
        """
        Sets up the next episode to be executed by creating required
        directories in the EXY working directory. The episode counter is
        increased and the `current_stats` file to write to updated
        accordingly.
        """
        self.episode_num += 1
        self.episode_ser = generate_episode_serial(self.episode_num)
        log.debug('Starting new episode: %s', self.episode_ser)

        ep_dir = self.get_episode_dir(self.episode_ser)
        snap_dir = get_snap_dir(ep_dir)
        ensure_directories(ep_dir, snap_dir)

        self.env.inp_dir = ep_dir
        self.env.snp_dir = snap_dir

        self.current_stats = get_stats_file(ep_dir)

    def setup(self):
        """
        Setups up the agent instance and output directories based on the given
        environment.
        """
        actions = self.env.action_space.shape[0]
        input_shape = self.env.observation_space.shape
        agent = create_agent(actions, input_shape)

        ensure_directories(self.exy_dir)

        self.load_brain(agent)
        self.load_properties()

        self.setup_next_episode()

        return agent

    def on_step_end(self, step, logs=None):
        """
        Called after each step performed during training. Fetches various
        statistics from the DoDonBotchi environment and writes them to the
        current episode's stats file.
        """
        row = [
            self.env.current_frame,
            self.env.current_lives,
            self.env.current_bombs,
            self.env.current_score,
            self.env.current_combo,
            self.env.current_grade,
            self.env.current_reward,
            self.env.current_hit,
            self.env.reward_sum,
            len(self.env.current_observation['enemies']),
            len(self.env.current_observation['bullets']),
            len(self.env.current_observation['ownshot']),
            len(self.env.current_observation['bonuses']),
            len(self.env.current_observation['powerup'])
        ]
        row = [str(c) for c in row]
        row = ';'.join(row)
        log.debug('Got step statistics: %s', row)
        row += '\n'

        with open(self.current_stats, 'a') as out_file:
            out_file.write(row)

        # ^C on Windows doesn't quit keras-rl's agent. Instead, we expect the
        # User to create a stop file and raise the KeyboardInterrupt
        # manually.                   v_v
        stop_file = self.get_stop_file()
        if os.path.exists(stop_file):
            os.remove(stop_file)
            raise KeyboardInterrupt() # lol

    def on_episode_end(self, episode, logs=None):
        """
        Called after each episode finishes. Fetches the last episode's score
        and enters them into EXY's leaderboard, sorted by score.
        """
        entry = {'ep': self.episode_ser, 'score': self.env.current_score}
        log.debug('Episode %s ended with score: %s', self.episode_ser,
                  self.env.current_score)
        self.leaderboard.append(entry)
        self.leaderboard = sorted(self.leaderboard, key=lambda e: e['score'])
        self.leaderboard.reverse()
        if len(self.leaderboard) > 32:
            self.leaderboard = self.leaderboard[:32]

        self.setup_next_episode()
        self.save_properties()
        log.debug('Finished an episode. Feel free to quit training with ^C.')

    def train(self):
        """
        Creates a DoDonPachi environment and a reinforcement learning agent for
        it and then trains it on that environment. Training can be interrupted
        and after training, either manually stopped or regularly terminated,
        current weights will be saved to EXY's brain file.
        """
        self.env = DoDonPachiEnv()
        agent = self.setup()
        log.info('Starting main training loop.')
        try:
            self.env.reset()
            agent.fit(self.env, callbacks=[self], nb_steps=STEPS, verbose=2)
        finally:
            self.env.close()
            self.save_brain(agent)
