local json = require('json')

local ctrl = nil
local state = nil
local ipc = nil

local screen = nil

local tickRate = {{tick_rate}}
local sleepFrames = 15

local cooldown = 0
local waitScore = false

function produceSocketOutput()
  local currentState = state.readGameState()
  local message = {message = 'gamestate', state = currentState}
  message = json.stringify(message)

  ipc.sendMessage(message)
end

function handleSocketInput()
  local message = ipc.readMessage()
  if message ~= nil then
    if message['command'] == 'wait' then
      emu.unpause()
      cooldown = tonumber(message['frames'])
    end

    if message['command'] == 'waitScore' then
      emu.unpause()
      waitScore = true
    end

    if message['command'] == 'kill' then
      manager:machine():exit()
    end

    if message['command'] == 'action' then
      ctrl.performAction(message['inputs'])
      emu.unpause()
      sleepFrames = tickRate
    end

    if message['command'] == 'snap' then
      screen:snapshot()
      ipc.sendACK()
    end

    if message['command'] == 'save' then
      local name = message['name']
      manager:machine():save(name)
      emu.pause()
      ipc.sendACK()
      cooldown = 2
    end

    if message['command'] == 'load' then
      local name = message['name']
      manager:machine():load(name)
      emu.pause()
      ctrl.performAction('0000')
      ipc.sendACK()
      cooldown = 2
    end
  end
end

function update()
  if cooldown > 0 then
    cooldown = cooldown - 1
    if cooldown <= 0 then
      sleepFrames = 1
    end
    return
  end

  if waitScore then
    local currentState = state.readGameState()
    if not currentState.scoreScreen then
      waitScore = false
      sleepFrames = 1
    end
    return
  end

  if manager:machine().paused then
    handleSocketInput()
  end

  if not manager:machine().paused then
    sleepFrames = sleepFrames - 1
    ctrl.updateInputStates()
    if sleepFrames == 0 then
      produceSocketOutput()
      emu.pause()
    else
    end
  end
end

function update_post()
end

function init(controller, gameState, comm)
  ctrl = controller
  state = gameState
  ipc = comm

  emu.register_frame(update)
  emu.register_frame_done(update_post)

  screen = manager:machine().screens[':screen']

  emu.pause()
end

local exports = {}

exports.init = init

return exports
