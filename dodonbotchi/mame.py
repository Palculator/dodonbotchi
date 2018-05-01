import logging as log
import socketserver
import os
import socket
import subprocess

from time import sleep

from jinja2 import Environment, FileSystemLoader
from dodonbotchi.config import CFG as cfg

PLUGIN_IPC = 'dodonbotchi_ipc'

TEMPLATE_LUA = 'init.lua'
TEMPLATE_JSON = 'plugin.json'


class MIPC:
    def __init__(self, con, addr):
        self.con = con
        self.addr = addr
        self.sfile = con.makefile(mode='rw', buffering=True)

    def read_state(self):
        pass

    def send_inputs(self):
        pass


def render_plugin(plugin_name, lua_name, json_name, **options):
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
    opts = {
        'host': cfg.host,
        'port': cfg.port,
        'show_sprites': cfg.show_sprites,
        'tick_rate': cfg.tick_rate
    }

    return render_plugin(PLUGIN_IPC, TEMPLATE_LUA, TEMPLATE_JSON, **opts)


def get_plugin_path(plugin_name):
    plugin_path = cfg.mame_path
    plugin_path = os.path.join(plugin_path, 'plugins')
    plugin_path = os.path.join(plugin_path, plugin_name)
    return plugin_path


def write_plugin(plugin_name, lua_code, json_code):
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
    lua_code, json_code = render_ipc_plugin()
    write_plugin(PLUGIN_IPC, lua_code, json_code)


def start_mame_ipc(video=False, audio=False, throttle=True, recording=None):
    write_ipc_plugin()

    socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_server.bind((cfg.host, cfg.port))
    socket_server.listen()
    log.info('Starting socket server on %s:%s', cfg.host, cfg.port)

    windowed = cfg.windowed

    call = []
    call.append('mame')
    call.append('-skip_gameinfo')
    call.append('-plugin')
    call.append(PLUGIN_IPC)
    if windowed:
        call.append('-window')
    if not video:
        call.append('-video')
        call.append('none')
    if not audio:
        call.append('-sound')
        call.append('none')
    if not throttle:
        call.append('-nothrottle')

    if cfg.save_state:
        call.append('-state')
        call.append(cfg.save_state)

    if recording:
        # TODO: Allow for custom recordings directory
        call.append('-record', recording)

    call.append('ddonpach')
    log.debug('Ended up with mame command: %s', call)

    process = subprocess.Popen(call)
    log.info('Started MAME with dodonbotchi ipc & dodonpachi.')
    log.info('Waiting for MAME to connect...')

    con, addr = socket_server.accept()
    mipc = MIPC(con, addr)
    log.info('Got mame connection from: %s', addr)
    return mipc
