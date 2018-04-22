local exports = {}
exports.name = 'dodonbotchi_api'
exports.version = '0.1'
exports.description = 'DoDonBotchi API'
exports.license = 'MIT'
exports.author = { name = 'Signaltonsalat' }

local botchi = exports

local debugger = nil
local consolelog = nil
local cpu = nil
local breaks = {byaddr = {}, byidx = {}}
local watches = {byaddr = {}, byidx = {}}

local socket = emu.file('', 7)
local connected = false

local running = false

function startBotchi()
    debugger = manager:machine():debugger()
    if debugger then
        consolelog = debugger.consolelog
    end

    cpu = manager:machine().devices[':maincpu']

    socket:open('socket.{{host}}:{{port}}')
end

function stopBotchi()
    running = false
    debugger = nil
    consolelog = nil
    cpu = nil
end

function updateBotchi()
    -- TODO: Everything
end

function botchi.startplugin()
    emu.register_start(startBotchi)
    emu.register_stop(stopBotchi)

    emu.register_frame_done(updateBotchi)
end

return exports

