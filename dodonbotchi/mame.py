"""
This module implements classes related to giving access to MAME and DoDonPachi
as an OpenAI-Gym-like environment.
"""
import json
import logging as log
import os
import random
import socket
import subprocess

from time import sleep

import numpy as np

from jinja2 import Environment, FileSystemLoader
from dodonbotchi.config import CFG as cfg
from dodonbotchi.util import get_now_string, ensure_directories
from rl.core import Env, Space

SPRITE_COUNT = 1024
OBSERVATION_DIMENSION = 1 + 1 + 1 + (SPRITE_COUNT * 5)

PLUGIN_IPC = 'dodonbotchi_ipc'
PLUGIN_REC = 'dodonbotchi_rec'

TEMPLATE_LUA = 'init.lua'
TEMPLATE_JSON = 'plugin.json'


class DoDonPachiActions(Space):
    """
    This class defines the possible action space for the DoDonPachi AI. Actions
    are encoded as a sequence of button-hold states for each possible button
    player 1 can press. The buttons are:

        UDLR123S

    Which represent up, down, left, right, button 1, button 2, button 3, and
    start. The hold-state of a button is given as either -, 0, or 1, which
    stand for:

        -: No change
        0: Stop holding
        1: Start holding

    An example action would be:

        ---11---

    Which would specify that player 1 should start holding right and button 1
    to shoot. If they wanted to stop shooting and also start moving up, the
    next action would be:

        1---0---

    Making player 1 start holding up, not touch the hold-state of left, but
    stop holding button 1.
    """

    def __init__(self, seed=None):
        self.buttons = 'UDLR12'
        self.states = '-01'
        self.count = len(self.states) ** len(self.buttons)
        self.rng = random.Random()
        if seed:
            self.rng.seed(seed)

    def from_ordinal(self, ordinal):
        """
        Converts an ordinal integer representing an action to an action encoded
        in the string format outlined above.
        """
        if ordinal >= self.count:
            raise ValueError('Invalid action ordinal: {}'.format(ordinal))

        action = ''
        for _ in self.buttons:
            idx = ordinal % len(self.states)
            action += self.states[idx]
            ordinal //= len(self.states)

        assert len(action) == len(self.buttons)
        return action

    def to_ordinal(self, action):
        """
        Converts a string-encoded action to an ordinal integer.
        """
        ordinal = 0
        for state in action[::-1]:
            idx = self.states.index(state)
            ordinal += idx
            ordinal *= len(self.states)

        assert ordinal < self.count
        return ordinal

    def sample(self, seed=None):
        """
        Creates a random DoDonPachi input action.
        """
        if seed:
            self.rng.seed(seed)

        ret = ''
        for _ in self.buttons:
            state = self.rng.choice(self.states)
            ret += state
        return ret

    def contains(self, action):
        """
        Tests if the given action string is a member of this action space and
        returns True iff it is. Criteria are:
            - Action is a string
            - Length is equal to the length of `BUTTONS`
            - Each character must be a member of `STATES`
        """
        if not isinstance(action, str):
            return False
        if len(action) != len(self.buttons):
            return False

        for state in action:
            if state not in self.states:
                return False

        return True


def observation_from_dict(dic):
    """
    Converts a game state dictionary to a high-dimensional tuple representing
    an observation for a neural network. The dictionary is expected to have
    the following items:
        * `bombs`: Current amount of bombs
        * `lives`: Current amount of lives left
        * `score`: Current score
        * `sprites`: A list of dictionaries representing a sprite, each one has:
            * `sid`: The sprite ID
            * `pos_x`: X position
            * `pos_y`: Y position
            * `siz_x`: Width
            * `siz_y`: Height

    The sprites list can at most have `SPRITE_COUNT` entries. More than that
    will cause a error, fewer than that will be padded with null sprites.

    The returned tuple has the following layout:

        (bombs, lives, score, sprite_1, sprite_2, ... sprite_n)

    Where each sprite's data is flatly written as five numbers, not a nested
    tuple. The observation is always a flat tuple of length
    `OBSERVATION_DIMENSION`.
    """

    if len(dic['sprites']) > SPRITE_COUNT:
        raise ValueError('Too many sprites: {}'.format(len(dic['sprites'])))

    bombs = dic['bombs']
    lives = dic['lives']
    score = dic['score']

    sprites = []
    for sprite_dic in dic['sprites']:
        sid = sprite_dic['sid']

        pos_x = sprite_dic['pos_x']
        pos_y = sprite_dic['pos_y']
        siz_x = sprite_dic['siz_x']
        siz_y = sprite_dic['siz_y']

        sprite = [sid, pos_x, pos_y, siz_x, siz_y]
        sprites.append(sprite)

    if len(sprites) < SPRITE_COUNT:
        for _ in range(SPRITE_COUNT - len(sprites)):
            sprite = [0, 0, 0, 0, 0]
            sprites.append(sprite)

    assert len(sprites) == SPRITE_COUNT

    sprites = sorted(sprites, key=lambda sprite: (sprite[1], sprite[2]))
    sprites = [n for sprite in sprites for n in sprite]

    observation = (bombs, lives, score, *sprites)
    assert len(observation) == OBSERVATION_DIMENSION
    return observation


class DoDonPachiObservations(Space):
    """
    Defines the observation space for DoDonPachi, an observation being a tuple
    describing the game state. The layout of such an observation is described
    in `observation_from_dict` and elements of this space are expected to
    follow it.
    """

    def __init__(self, seed=None):
        self.rng = random.Random()
        if seed:
            self.rng.seed(seed)

        self.dimension = OBSERVATION_DIMENSION

        self.max_bombs = 6
        self.max_lives = 3
        self.max_score = 99999999
        self.max_pos = 4095
        self.max_siz = 511

    def sample(self, seed=None):
        """
        Procues a random observation valid according to the format laid out in
        the documentation of `observation_from_dict` and returns it as a tuple.
        """
        if seed:
            self.rng.seed(seed)

        bombs = self.rng.randint(0, self.max_bombs)
        lives = self.rng.randint(0, self.max_lives)
        score = self.rng.randint(0, self.max_score)

        sprites = []
        for _ in range(SPRITE_COUNT):
            sid = self.rng.randint(0, 2 ** 32)

            pos_x = self.rng.randint(0, self.max_pos)
            pos_y = self.rng.randint(0, self.max_pos)

            siz_x = self.rng.randint(0, self.max_siz)
            siz_y = self.rng.randint(0, self.max_siz)

            sprite = {
                'sid': sid,
                'pos_x': pos_x,
                'pos_y': pos_y,
                'siz_x': siz_x,
                'siz_y': siz_y
            }
            sprites.append(sprite)

        dic = dict()
        dic['bombs'] = bombs
        dic['lives'] = lives
        dic['score'] = score
        dic['sprites'] = sprites

        return observation_from_dict(dic)

    def contains(self, observation):
        """
        Tests if an observation is contained in DoDonPachi's observation space
        and returns True iff it is. The criteria to be considered contained
        are:

            - Be a tuple
            - Be of length `OBSERVATION_DIMENSION`
            - Follow the observation layout described by `observation_from_dict`
            - Bomb count >= 0 and < `max_bombs`
            - Live count >= 0 and < `max_lives`
            - Score >= 0 and < `max_score`
            - Each sprite's properties must follow:
                - `sid` >= 0 and < 2 ** 32
                - `pos_x` and `pos_y` >= 0 and < `max_pos`
                - `siz_x` and `siz_y` >= 0 and < `max_siz`
        """

        if not isinstance(observation, tuple):
            return False

        if len(observation) != OBSERVATION_DIMENSION:
            return False

        bombs = observation[0]
        if bombs < 0 or bombs > self.max_bombs:
            return False

        lives = observation[1]
        if lives < 0 or lives > self.max_lives:
            return False

        score = observation[2]
        if score < 0 or score > self.max_score:
            return False

        okay = True

        sprites = observation[3:]
        for idx in range(0, SPRITE_COUNT * 5, 5):
            sid, pos_x, pos_y, siz_x, siz_y = sprites[idx:idx+5]

            if sid < 0 or sid >= 2 ** 32:
                okay = False
                break

            if pos_x < 0 or pos_x > self.max_pos:
                okay = False
                break

            if pos_y < 0 or pos_y > self.max_pos:
                okay = False
                break

            if siz_x < 0 or siz_x > self.max_siz:
                okay = False
                break

            if siz_y < 0 or siz_y > self.max_siz:
                okay = False
                break

        return okay


def render_plugin(plugin_name, lua_name, json_name, **options):
    """
    Renders a MAME plugin from the templates in the directory identified by the
    given plugin_name. For the lua code, the template with the name lua_name
    will be used and for the json-properties, the template json_name will be
    used. The templates will be rendered using the given **options as values.

    The resulting lua and json code is returned as a tuple (lua, json).
    """
    lua_name = os.path.join(plugin_name, lua_name)
    json_name = os.path.join(plugin_name, json_name)

    template_env = Environment(loader=FileSystemLoader('dodonbotchi/plugin'))
    template_lua = template_env.get_template(lua_name)
    template_json = template_env.get_template(json_name)

    options['plugin_name'] = plugin_name

    lua_code = template_lua.render(**options)
    json_code = template_json.render(**options)

    return lua_code, json_code


def render_ipc_plugin():
    """
    Renders the IPC plugin used to communicate with MAME using configuration
    values from the global config. The rendered code is returned as a (lua,
    json) tuple.
    """
    opts = {
        'host': cfg.host,
        'port': cfg.port,
        'show_sprites': cfg.show_sprites,
        'tick_rate': cfg.tick_rate,
        'snap_rate': cfg.snap_rate
    }

    return render_plugin(PLUGIN_IPC, TEMPLATE_LUA, TEMPLATE_JSON, **opts)


def render_rec_plugin():
    """
    Renders the recording display plugin used to display information while
    rendering out recordings of AI trials. The rendered code is returned as a
    (lua, json) tuple.
    """
    opts = {}
    return render_plugin(PLUGIN_REC, TEMPLATE_LUA, TEMPLATE_JSON, **opts)


def get_plugin_path(plugin_name):
    """
    Gets the target directory to save a plugin of the given name to, using the
    MAME home directory specified in the global config.
    """
    plugin_path = cfg.mame_path
    plugin_path = os.path.join(plugin_path, 'plugins')
    plugin_path = os.path.join(plugin_path, plugin_name)
    return plugin_path


def write_plugin(plugin_name, lua_code, json_code):
    """
    Saves a plugin of the given name using the given lua code and json
    properties to the MAME home directory specified in the global config.
    """
    plugin_path = get_plugin_path(plugin_name)
    if not os.path.exists(plugin_path):
        os.makedirs(plugin_path)

    lua_file = TEMPLATE_LUA
    lua_file = os.path.join(plugin_path, lua_file)
    with open(lua_file, 'w') as out_file:
        out_file.write(lua_code)

    json_file = TEMPLATE_JSON
    json_file = os.path.join(plugin_path, json_file)
    with open(json_file, 'w') as out_file:
        out_file.write(json_code)


def write_ipc_plugin():
    """
    Renders and saves the IPC plugin to the MAME home directory specified in
    the global config.
    """
    lua_code, json_code = render_ipc_plugin()
    write_plugin(PLUGIN_IPC, lua_code, json_code)


def write_rec_plugin():
    """
    Renders and saves the recording display plugin to the MAME home directory
    specified in the global config.
    """
    lua_code, json_code = render_rec_plugin()
    write_plugin(PLUGIN_REC, lua_code, json_code)



def generate_base_call():
    """
    Generates a list of parameters to start MAME with DoDonPachi which contain
    command line options corresponding to what's configured in the global
    configuration. The list gets returned and can be customised by appending
    further options.
    """
    call = ['mame', 'ddonpach', '-skip_gameinfo']

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
    write_rec_plugin()

    call = generate_base_call()

    call.append('-plugin')
    call.append(PLUGIN_REC)

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
        self.reward_range = (-1, 99999999)
        self.action_space = DoDonPachiActions(seed)
        self.observation_space = DoDonPachiObservations(seed)

        self.inp_dir = None
        self.snp_dir = None

        self.recording = None

        self.process = None
        self.server = None
        self.client = None
        self.sfile = None

        self.current_lives = 0
        self.current_score = 0

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((cfg.host, cfg.port))
        self.server.listen()
        log.info('Started socket server on %s:%s', cfg.host, cfg.port)

        write_ipc_plugin()

    def configure(self, **options):
        """
        Configures this environment. Expected options are:
            inp_dir: Directory to save .inp recordings to
            snp_dir: Directory to save snapshots to
        """
        if 'inp_dir' not in options:
            raise ValueError('Config needs to specify inp_dir.')
        self.inp_dir = options.get('inp_dir')

        if 'snp_dir' not in options:
            raise ValueError('Config needs to specify snp_dir.')
        self.snp_dir = options.get('snp_dir')

        ensure_directories(self.inp_dir, self.snp_dir)

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

    def read_observation(self):
        """
        Reads an observation from the client, ensuring it's a member of our
        observation space and converting it to a tuple as described in
        `observation_from_dict` before returning it.
        """
        message = self.read_message()
        assert 'message' in message and message['message'] == 'observation'
        observation_dic = message['observation']
        observation = observation_from_dict(observation_dic)
        assert self.observation_space.contains(observation)
        return observation

    def start_mame(self):
        """
        Boots up MAME with the globally configured options and additionally
        setting it to record inputs and screenshots to this environment's
        respective folders for those.
        """
        call = generate_base_call()
        call.append('-plugin')
        call.append(PLUGIN_IPC)

        abs_inp_dir = os.path.abspath(self.inp_dir)
        call.append('-input_directory')
        call.append(abs_inp_dir)
        call.append('-record')
        call.append(self.recording)

        abs_snp_dir = os.path.join(self.snp_dir, get_now_string())
        abs_snp_dir = os.path.abspath(abs_snp_dir)
        call.append('-snapshot_directory')
        call.append(abs_snp_dir)

        self.process = subprocess.Popen(call)
        log.info('Started MAME with dodonbotchi ipc & dodonpachi.')
        log.info('Waiting for MAME to connect...')

        self.client, addr = self.server.accept()
        self.sfile = self.client.makefile(mode='rw', buffering=True)
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
        observation = self.read_observation()

        lives = observation[1]
        score = observation[2]

        if lives < self.current_lives:
            reward = -1
        else:
            reward = score - self.current_score

        done = lives == 0

        self.current_lives = lives
        self.current_score = score

        return observation, reward, done, {}

    def reset(self):
        """
        Resets MAME and DoDonPachi to start from scratch. The initial
        observation immediately after starting the game is returned.
        """
        self.current_lives = 0
        self.current_score = 0

        self.recording = '{}.inp'.format(get_now_string())

        if self.process:
            self.stop_mame()

        self.start_mame()

        observation = self.read_observation()
        self.current_lives = observation[0]
        self.current_score = observation[2]
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
