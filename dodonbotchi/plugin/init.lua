local exports = {}
exports.name = '{{plugin_name}}'
exports.version = '0.1'
exports.description = 'DoDonBotchi API'
exports.license = 'MIT'
exports.author = { name = 'Signaltonsalat' }

local botchi = exports

local debugger = nil
local consolelog = nil
local cpu = nil
local mem = nil
local breaks = {byaddr = {}, byidx = {}}
local watches = {byaddr = {}, byidx = {}}

local socket = nil
local connected = false
local running = false

function sendMessage(message)
    socket:write(message..'\n')
end

function readGameState()
    ret = {}
    ret['bombs'] = mem:read_u8(0x102CB0)
    return ret
end

function startBotchi()
    debugger = manager:machine():debugger()
    if debugger then
        consolelog = debugger.consolelog
    end

    cpu = manager:machine().devices[':maincpu']
    mem = cpu.spaces['program']

    sendMessage('HELLO')
end

function stopBotchi()
    running = false
    debugger = nil
    consolelog = nil
    cpu = nil
end

function updateBotchi()
    if connected then
        local state = readGameState()
        socket:write(sendMessage(tostring(state['bombs'])))
    end
end

function botchi.startplugin()
    socket = emu.file('rw')
    socket:open('socket.{{host}}:{{port}}')
    connected = true

    emu.register_start(startBotchi)
    emu.register_frame(updateBotchi)
    emu.register_stop(stopBotchi)
end

return exports

