import logging as l
import socketserver
import os
import subprocess
import socket

from jinja2 import Environment, FileSystemLoader
from dodonbotchi.config import cfg

TEMPLATE_LUA = 'init.lua'
TEMPLATE_JSON = 'plugin.json'


def render_plugin():
    template_env = Environment(loader=FileSystemLoader('dodonbotchi/plugin'))
    template_lua = template_env.get_template(TEMPLATE_LUA)
    template_json = template_env.get_template(TEMPLATE_JSON)

    options = {
        'host': cfg.host,
        'port': cfg.port,
        'plugin_name': cfg.plugin_name
    }

    lua_code = template_lua.render(**options)
    json_code = template_json.render(**options)

    return lua_code, json_code


def get_plugin_path():
    plugin_path = cfg.mame_path
    plugin_path = os.path.join(plugin_path, 'plugins')
    plugin_path = os.path.join(plugin_path, cfg.plugin_name)
    return plugin_path


def write_plugin():
    plugin_path = get_plugin_path()
    if not os.path.exists(plugin_path):
        os.makedirs(plugin_path)

    lua_code, json_code = render_plugin()

    lua_file = os.path.join(plugin_path, TEMPLATE_LUA)
    with open(lua_file, 'w') as out_file:
        out_file.write(lua_code)

    json_file = os.path.join(plugin_path, TEMPLATE_JSON)
    with open(json_file, 'w') as out_file:
        out_file.write(json_code)


class MAPI(socketserver.StreamRequestHandler):
    def read_line(self):
        line = self.rfile.readline()
        line = line[:-1]
        line = str(line, 'ascii')
        return line

    def handle(self):
        line = self.read_line
        while line != 'HELLO':
            line = self.read_line()

        while line and line != 'GOODBYE':
            line = self.read_line()
            l.debug('Got line from client: %s', line)


def run_mame():
    write_plugin()

    server = socketserver.TCPServer((cfg.host, cfg.port), MAPI)
    l.info('Starting socket server on %s:%s', cfg.host, cfg.port)
    call = ['mame', '-window', '-plugin', cfg.plugin_name, 'ddonpach']
    process = subprocess.Popen(call)
    l.info('Started MAME with dodonbotchi & dodonpachi.')
    while True:
        server.handle_request()
