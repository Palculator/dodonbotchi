local exports = {}
exports.name = '{{plugin_name}}'
exports.version = '1.0'
exports.description = 'DoDonBotchi MAME Component'
exports.license = 'MIT'
exports.author = {name = 'Signaltonsalat'}

local cpu = nil
local mem = nil
local screen = nil

local ctrl = nil
local sprt = nil
local state = nil

local renderState = {{render_state}}
local renderSprites = {{render_sprites}}
local showInput = {{show_input}}

function startBotchi()
  cpu = manager:machine().devices[':maincpu']
  mem = cpu.spaces['program']
  screen = manager:machine().screens[':screen']
  
  ctrl = require('{{plugin_name}}/controller')
  sprt = require('{{plugin_name}}/sprites')
  state = require('{{plugin_name}}/state')
  
  ctrl.init()
  state.init(sprt)
  
  startRC()
end

function startRC()
  local ipc = require('{{plugin_name}}/ipc')
  ipc.init()
  
  rc = require('{{plugin_name}}/remoteController')
  rc.init(ctrl, state, ipc)
end

function displayBotchi()
  if renderState then
    state.readGameState()
    state.render()
  end
  
  if renderSprites then
    sprt.readSprites(mem, 0)
    sprt.render(screen)
  end
  
  if showInput then
    ctrl.render(screen)
  end
end

function startPlugin()
  emu.register_start(startBotchi)
  emu.register_frame_done(displayBotchi)
end

exports.startplugin = startPlugin

return exports

