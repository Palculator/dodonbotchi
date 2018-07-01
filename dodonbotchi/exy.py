"""
This module contains DoDonBotchi's brain and neuroevolutional code. It handles
learning, evolution, and controlling DoDonPachi in realtime.
"""
import json
import logging as log
import os
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

from . import agents

from .config import CFG as cfg
from .mame import Ddonpach, DoDonPachiEnv
from .util import generate_now_serial_number
from .util import ensure_directories


EPS_DIR = 'episodes'

BRAIN_FILE = 'brain.h5f'
PROPS_FILE = 'exy.json'
STATS_FILE = 'stats.csv'
STOP_FILE = 'stop.pls'

SNAP_DIR = 'snap'

WARMUP_INIT = 50000
STEPS = 1000000

COLS = [
    'Episode',
    'Frame',
    'Lives',
    'Bombs',
    'Score',
    'Combo',
    'Reward',
    'Hit',
    'RewardSum',
    'Enemies',
    'Bullets',
    'Ownshot',
    'Bonuses',
    'PowerUp'
]


def get_snap_dir(ep_dir):
    return os.path.join(ep_dir, SNAP_DIR)


def get_stats_file(ep_dir):
    return os.path.join(ep_dir, STATS_FILE)


def generate_episode_serial(ep_num):
    serial = generate_now_serial_number()
    serial = '{:08} - {}'.format(ep_num, serial)
    return serial


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

    def __init__(self, exy_dir, agent_name):
        self.exy_dir = exy_dir
        self.agent_name = agent_name

        self.ddonpach = None
        self.env = None

        self.warmup = WARMUP_INIT

        self.episode_num = 0
        self.episode_ser = None
        self.episode_dir = None
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
            self.warmup = props_dic['warmup']
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
            props_dic['warmup'] = self.warmup
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

        self.episode_dir = self.get_episode_dir(self.episode_ser)
        snap_dir = get_snap_dir(self.episode_dir)
        ensure_directories(self.episode_dir, snap_dir)

        self.ddonpach.inp_dir = self.episode_dir
        self.ddonpach.snp_dir = snap_dir

        self.current_stats = get_stats_file(self.episode_dir)

    def setup(self):
        """
        Setups up the agent instance and output directories based on the given
        environment.
        """
        actions = self.env.action_space.shape[0]

        agent = agents.create_agent(self.agent_name,
                                    self,
                                    self.env,
                                    actions)

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
            self.episode_num,
            self.env.current_frame,
            self.env.current_lives,
            self.env.current_bombs,
            self.env.current_score,
            self.env.current_combo,
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
            raise KeyboardInterrupt()  # lol

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
        self.ddonpach = Ddonpach()
        self.env = DoDonPachiEnv(self.ddonpach)
        agent = self.setup()
        log.info('Starting main training loop.')
        try:
            self.env.reset()
            agent.fit(self.env, callbacks=[self], nb_steps=STEPS, verbose=2)
        finally:
            self.env.close()
            self.save_brain(agent)

    def test(self, episodes):
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
            agent.test(self.env, nb_episodes=episodes)
        finally:
            self.env.close()

    def plot_lm(self, plot_file, stats, x, y):
        plot = sns.lmplot(x=x, y=y, data=stats,
                          palette='muted', scatter_kws={'s': 0.1})
        plot.savefig(plot_file, dpi=300)
        log.info('Plotted: %s', plot_file)
        plt.close('all')

    def plot_overall(self):
        overall = []
        total_steps = 0

        episodes_dir = self.get_episodes_dir()
        episodes = sorted(os.listdir(episodes_dir))
        for episode in episodes:
            episode_dir = os.path.join(episodes_dir, episode)
            stats_file = get_stats_file(episode_dir)
            with open(stats_file, 'r') as in_file:
                stats = in_file.readlines()
            steps = len(stats)
            total_steps += steps
            stats = stats[-1]
            stats = stats.split(';')
            stats = stats + [steps, total_steps]
            overall.append(stats)

        columns = COLS + ['Steps', 'TotalSteps']
        stats = pd.DataFrame(data=overall, columns=columns)
        stats[columns] = stats[columns].apply(pd.to_numeric, axis=1)

        score_episode = os.path.join(self.exy_dir, 'score_episode.png')
        self.plot_lm(score_episode, stats, 'Episode', 'Score')

        steps_episode = os.path.join(self.exy_dir, 'steps_episode.png')
        self.plot_lm(steps_episode, stats, 'Episode', 'Steps')

        total_steps_episode = os.path.join(self.exy_dir,
                                           'total_steps_episode.png')
        self.plot_lm(total_steps_episode, stats, 'Episode', 'TotalSteps')
