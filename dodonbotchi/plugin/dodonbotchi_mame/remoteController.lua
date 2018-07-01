local ctrl = nil
local state = nil
local ipc = nil

local tickRate = {{tick_rate}}
local sleepFrames = 15

function produceSocketOutput()
    local currentState = state.readGameState()
    local message = { message = 'observation', observation = currentState }
    message = json.stringify(message)

    ipc.sendMessage(message)
end

function handleSocketInput()
    local message = ipc.readMessage()

    if message ~= nil then
        if message['command'] == 'kill' then
            manager:machine():exit()
        end

        if message['command'] == 'action' then
            ctrl.performAction(message['inputs'])
            emu.unpause()
            sleepFrames = tickRate
        end
    end
end

function update()
    if manager:machine().paused then
        handleSocketInput()
    end

    if not manager:machine().paused then
        ctrl.updateInputStates()
    end
end

function update_post()
    produceSocketOutput()
    emu.pause()
end

function init(controller, gameState, comm)
    ctrl = controller
    state = gameState
    ipc = comm

    emu.register_frame(update)
    emu.register_frame_done(update_post)
end

local exports = {}

exports.init = init

return exports
