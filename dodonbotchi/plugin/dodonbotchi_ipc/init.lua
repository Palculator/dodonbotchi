local json = require('json')

local exports = {}
exports.name = '{{plugin_name}}'
exports.version = '0.1'
exports.description = 'DoDonBotchi IPC'
exports.license = 'MIT'
exports.author = {name = 'Signaltonsalat'}

local botchi = exports

local cpu = nil
local mem = nil
local screen = nil

local inputMap = {}
local inputState = {}
local buttons = 'UDLR12'

local socket = nil
local connected = false

local tickRate = {{tick_rate}}
local snapRate = {{snap_rate}}
local sleepFrames = 30

local showSprites = {{show_sprites}}
local spriteColours = {
    0xFFFF0000,
    0xFF00FF00,
    0xFF0088FF
}
local sprites = {}

function initInputMap()
    for tag, port in pairs(manager:machine():ioport().ports) do
        if port.fields['P1 Up'] then
            inputMap['U'] = port.fields['P1 Up']
            inputState['U'] = 0
        end

        if port.fields['P1 Down'] then
            inputMap['D'] = port.fields['P1 Down']
            inputState['D'] = 0
        end

        if port.fields['P1 Left'] then
            inputMap['L'] = port.fields['P1 Left']
            inputState['L'] = 0
        end

        if port.fields['P1 Right'] then
            inputMap['R'] = port.fields['P1 Right']
            inputState['R'] = 0
        end

        if port.fields['P1 Button 1'] then
            inputMap['1'] = port.fields['P1 Button 1']
            inputState['1'] = 0
        end

        if port.fields['P1 Button 2'] then
            inputMap['2'] = port.fields['P1 Button 2']
            inputState['2'] = 0
        end

        if port.fields['P1 Button 3'] then
            inputMap['3'] = port.fields['P1 Button 3']
            inputState['3'] = 0
        end

        if port.fields['1 Player Start'] then
            inputMap['S'] = port.fields['1 Player Start']
            inputState['S'] = 0
        end

        if port.fields['Coin 1'] then
            inputMap['C'] = port.fields['Coin 1']
            inputState['C'] = 0
        end
    end
end

function updateInputs()
    for k, v in pairs(inputState) do
        if v == 1 then
            inputMap[k]:set_value(0)
            inputState[k] = 0
        end

        if v == -1 then
            inputMap[k]:set_value(1)
        end

        if v > 1 then
            inputState[k] = inputState[k] - 1
        end
    end
end

function startHold(button)
    inputMap[button]:set_value(1)
    inputState[button] = -1
end

function stopHold(button)
    inputMap[button]:set_value(0)
    inputState[button] = 0
end

function singlePress(button)
    inputMap[button]:set_value(1)
    inputState[button] = tickRate
end

function readScore()
    -- The score is in a weird format where each digit of the hex-encoded
    -- number is the digit of the score in its decimal form. 0x11814 = 11814
    -- Final number is multiplied by 10 because the last digit of the score is
    -- the credit count.
    local weird = mem:read_u32(0x10161E)
    local score = weird % 16
    local power = 10
    for i = 0, 6 do
        weird = math.floor(weird / 16)
        score = score + (weird % 16) * power
        power = power * 10
    end
    return score * 10
end

function readGameState()
    readSprites()

    ret = {}
    ret['bombs'] = mem:read_u8(0x102CB0)
    ret['lives'] = mem:read_u8(0x101965)
    ret['score'] = readScore()
    ret['sprites'] = sprites

    return ret
end

function startBotchi()
    cpu = manager:machine().devices[':maincpu']
    mem = cpu.spaces['program']
    screen = manager:machine().screens[':screen']

    initInputMap()
end

function readSprites()
    local sid, sid1, sid2, pos, x, y, mode, width, height, data

    sprites = {}
    for i = 0x400000, 0x404000 - 0x10, 0x10 do
        sid = mem:read_u64(i + 0x0)
        pos = sid % 4294967296
        sid = math.floor(sid / 4294967296)

        if sid > 0 then
            pos_x = pos / 65536
            pos_y = pos % 65536

            mode = mem:read_u16(i + 0x8)

            siz_x = 16 * (mode / 256)
            siz_y = 16 * (mode % 256)

            data = {sid = sid,
                    pos_x = pos_x,
                    pos_y = pos_y,
                    siz_x = siz_x,
                    siz_y = siz_y}

            table.insert(sprites, data)
        end
    end
end

function displayBotchi()
    local pos, x, y, width, height, colour_idx, colour

    for i, v in pairs(sprites) do
        pos_x = v['pos_x']
        pos_y = v['pos_y']
        siz_x = v['siz_x']
        siz_y = v['siz_y']

        colour_idx = v['sid'] % 3 + 1
        colour = spriteColours[colour_idx]

        screen:draw_box(pos_x, pos_y, pos_x + siz_x, pos_y + siz_y, 0, colour)
    end
end

function sendMessage(message)
    socket:write(message .. '\n')
end

function produceSocketOutput()
    local state, message

    state = readGameState()
    message = { message = 'observation', observation = state }
    message = json.stringify(message)

    sendMessage(message)
end

function performAction(action)
    for i = 1, #buttons do
        local button = buttons:sub(i, i)
        local state = action:sub(i, i)

        if state == '1' then
            startHold(button)
        end

        if state == '0' then
            stopHold(button)
        end
    end
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

function updateBotchi()
    if connected then
        local paused = manager:machine().paused

        if screen:frame_number() % snapRate == 0 then
            screen:snapshot()
        end

        if paused then
            handleSocketInput()
        else

            updateInputs()

            if sleepFrames == 0 then
                produceSocketOutput()
                emu.pause()
            else
                sleepFrames = sleepFrames - 1
            end
        end
    end
end

function botchi.startplugin()
    socket = emu.file('rw')
    socket:open('socket.{{host}}:{{port}}')
    connected = true

    emu.register_start(startBotchi)
    emu.register_frame(updateBotchi)
    if showSprites then
        emu.register_frame_done(displayBotchi)
    end
end

return exports
