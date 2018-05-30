local json = require('json')

local exports = {}

local buttons = 'VH1'

local mem = nil
local screen = nil

local ctrl = nil
local state = nil

local socket = nil
local connected = false

local tickRate = {{tick_rate}}
local dumpFrames = {{dump_frames}}
local sleepFrames = 15

local currentState = {}

function sendMessage(message)
    socket:write(message .. '\n')
end

function produceSocketOutput()
    currentState = state.readGameState()
    local message = { message = 'observation', observation = currentState }
    message = json.stringify(message)

    sendMessage(message)
end

function performDirection(state, posKey, negKey)
    if state == '1' then
        ctrl.stopHold(posKey)
        ctrl.singlePress(negKey)
    end

    if state == '2' then
        ctrl.stopHold(negKey)
        ctrl.singlePress(posKey)
    end
end

function performButton(state, key)
    if state == '1' then
        ctrl.singlePress(key)
    end
end

function performAction(action)
    local vertical = action:sub(1, 1)
    performDirection(vertical, 'U', 'D')
    local horizontal = action:sub(2, 2)
    performDirection(horizontal, 'R', 'L')

    local button1 = action:sub(3, 3)
    performButton(button1, '1')
    -- Bombs disabled for now because the AI is stupid.
    -- local button2 = action:sub(4, 4)
    -- performButton(button2, '2')
end

function handleSocketInput()
    local message = socket:read(1024)

    if message ~= nil and #message > 0 then
        message = json.parse(message)
        if message['command'] == 'kill' then
            manager:machine():exit()
        end

        if message['command'] == 'action' then
            performAction(message['inputs'])
            emu.unpause()
            sleepFrames = tickRate
        end
    end
end

function update()
    if connected then
        if manager:machine().paused then
            handleSocketInput()
        end

        if not manager:machine().paused then
            ctrl.updateInputStates()

            if sleepFrames == 0 then
                if dumpFrames then
                    screen:snapshot()
                end
                produceSocketOutput()
                emu.pause()
            else
                sleepFrames = sleepFrames - 1
            end
        end
    end
end

function init(controller, gameState)
    ctrl = controller
    state = gameState

    mem = manager:machine().devices[':maincpu'].spaces['program']
    screen = manager:machine().screens[':screen']

    socket = emu.file('rw')
    socket:open('socket.{{host}}:{{port}}')
    connected = true

    emu.register_frame(update)
end

exports.init = init

return exports
