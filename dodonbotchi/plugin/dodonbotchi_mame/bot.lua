local json = require('json')

local exports = {}

local buttons = 'VH1'

local mem = nil
local screen = nil

local ctrl = nil
local state = nil
local ipc = nil

local tickRate = {{tick_rate}}
local dumpFrames = {{dump_frames}}
local renderRing = {{render_ring}}
local sleepFrames = 15

local currentState = {}

local shipX = 0
local shipY = 0
local enemiesRing = nil
local bulletsRing = nil

function produceSocketOutput()
    currentState = state.readGameState()
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

        if message['command'] == 'threat_ring' then
            shipX = message['ship_x']
            shipY = message['ship_y']
            enemiesRing = message['enemies']
            bulletsRing = message['bullets']
            ipc.sendACK()
        end
    end
end

function update()
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

function renderRing(ring, radius, red)
    local segs = #ring
    local step = 360 / segs
    local start = -step/2.0

    for i, v in ipairs(ring) do
        local beg_angle = (i - 1) * step - (step / 2.0)
        local end_angle = i * step - (step / 2.0)

        beg_angle = math.rad(beg_angle)
        end_angle = math.rad(end_angle)

        beg_x = radius * math.cos(beg_angle) + shipX
        beg_y = radius * math.sin(beg_angle) + shipY
        end_x = radius * math.cos(end_angle) + shipX
        end_y = radius * math.sin(end_angle) + shipY

        local colour = 1.0 - math.min(1.0, v / 4.0)
        colour = math.floor(255 * colour)

        if red then
            colour = 0xFFFF * 65536 + (colour * 256) + colour
        else
            colour = 0xFF * 16777216 + (colour * 65536) + (0xFF * 256) + colour
        end

        screen:draw_line(beg_x, beg_y, end_x, end_y, colour)
    end
end

function render()
    if enemiesRing then
        renderRing(enemiesRing, 18, false)
    end

    if bulletsRing then
        renderRing(bulletsRing, 22, true)
    end
end

function init(controller, gameState, comm)
    ctrl = controller
    state = gameState
    ipc = comm

    mem = manager:machine().devices[':maincpu'].spaces['program']

    emu.register_frame(update)

    screen = manager:machine().screens[':screen']
end

exports.init = init
exports.render = render

return exports
