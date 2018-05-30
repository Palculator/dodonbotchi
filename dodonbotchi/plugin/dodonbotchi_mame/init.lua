local exports = {}
exports.name = '{{plugin_name}}'
exports.version = '0.1'
exports.description = 'DoDonBotchi MAME Component'
exports.license = 'MIT'
exports.author = {name = 'Signaltonsalat'}

local cpu = nil
local mem = nil
local screen = nil

local ctrl = nil
local sprt = nil
local state = nil

local mode = '{{mode}}'
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

    if mode == 'bot' then
        startBot()
    end
end

function startBot()
    local bot = require('{{plugin_name}}/bot')
    bot.init(ctrl, state)
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

