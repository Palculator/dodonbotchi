local json = require('json')

local exports = {}
exports.name = '{{plugin_name}}'
exports.version = '0.1'
exports.description = 'DoDonBotchi Recording display'
exports.license = 'MIT'
exports.author = {name = 'Signaltonsalat'}

local botchi = exports

local bgColour = 0xFFBDBDD6
local pressedColour = 0xFFFF0000
local unpressedColour = 0xFF000000

local ctrlOriginX = 0
local ctrlOriginY = 164

local cpu = nil
local mem = nil
local screen = nil

local inputMap = {}
local buttonStates = {}

function initInputMap()
    for tag, port in pairs(manager:machine():ioport().ports) do
        if port.fields['P1 Up'] then
            inputMap['U'] = {port = port, field = port.fields['P1 Up']}
        end

        if port.fields['P1 Down'] then
            inputMap['D'] = {port = port, field = port.fields['P1 Down']}
        end

        if port.fields['P1 Left'] then
            inputMap['L'] = {port = port, field = port.fields['P1 Left']}
        end

        if port.fields['P1 Right'] then
            inputMap['R'] = {port = port, field = port.fields['P1 Right']}
        end

        if port.fields['P1 Button 1'] then
            inputMap['1'] = {port = port, field = port.fields['P1 Button 1']}
        end

        if port.fields['P1 Button 2'] then
            inputMap['2'] = {port = port, field = port.fields['P1 Button 2']}
        end

        if port.fields['P1 Button 3'] then
            inputMap['3'] = {port = port, field = port.fields['P1 Button 3']}
        end
    end
end

function readButtonState(key)
    local state, button

    button = inputMap[key]
    state = ((button.port:read() & button.field.mask) - button.field.defvalue)
    buttonStates[key] = state ~= 0
end

function readButtonStates()
    readButtonState('U')
    readButtonState('D')
    readButtonState('L')
    readButtonState('R')
    readButtonState('1')
    readButtonState('2')
    readButtonState('3')
end

function startBotchi()
    cpu = manager:machine().devices[':maincpu']
    mem = cpu.spaces['program']
    screen = manager:machine().screens[':screen']

    initInputMap()
end

function displayBotchi()
    local btnColour

    readButtonStates()

    screen:draw_box(ctrlOriginX, ctrlOriginY, ctrlOriginX + 16, ctrlOriginY + 36, bgColour, 0)
    
    btnColour = unpressedColour
    if buttonStates['U'] then
        btnColour = pressedColour
    end
    screen:draw_box(ctrlOriginX + 10, ctrlOriginY + 6, ctrlOriginX + 14, ctrlOriginY + 10, btnColour, 0)

    btnColour = unpressedColour
    if buttonStates['D'] then
        btnColour = pressedColour
    end
    screen:draw_box(ctrlOriginX + 2, ctrlOriginY + 6, ctrlOriginX + 6, ctrlOriginY + 10, btnColour, 0)

    btnColour = unpressedColour
    if buttonStates['L'] then
        btnColour = pressedColour
    end
    screen:draw_box(ctrlOriginX + 6, ctrlOriginY + 2, ctrlOriginX + 10, ctrlOriginY + 6, btnColour, 0)

    btnColour = unpressedColour
    if buttonStates['R'] then
        btnColour = pressedColour
    end
    screen:draw_box(ctrlOriginX + 6, ctrlOriginY + 10, ctrlOriginX + 10, ctrlOriginY + 14, btnColour, 0)

    btnColour = unpressedColour
    if buttonStates['1'] then
        btnColour = pressedColour
    end
    screen:draw_box(ctrlOriginX + 6, ctrlOriginY + 18, ctrlOriginX + 10, ctrlOriginY + 22, btnColour, 0)

    btnColour = unpressedColour
    if buttonStates['2'] then
        btnColour = pressedColour
    end
    screen:draw_box(ctrlOriginX + 6, ctrlOriginY + 24, ctrlOriginX + 10, ctrlOriginY + 28, btnColour, 0)

    btnColour = unpressedColour
    if buttonStates['3'] then
        btnColour = pressedColour
    end
    screen:draw_box(ctrlOriginX + 6, ctrlOriginY + 30, ctrlOriginX + 10, ctrlOriginY + 34, btnColour, 0)
end

function botchi.startplugin()
    emu.register_start(startBotchi)
    emu.register_frame_done(displayBotchi)
end

return exports
