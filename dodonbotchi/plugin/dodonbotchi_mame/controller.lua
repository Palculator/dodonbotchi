local exports = {}

local inputMap = {}
local inputStates = {}
local buttonStates = {}

local bgColour = 0xFFBDBDD6
local pressedColour = 0xFFFF0000
local unpressedColour = 0xFF000000

local ctrlOriginX = 0
local ctrlOriginY = 164

local tickRate = {{tick_rate}}

function init()
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

function updateInputStates()
    for k, v in pairs(inputStates) do
        if v == 1 then
            inputMap[k].field:set_value(0)
            inputStates[k] = 0
        end

        if v == -1 then
            inputMap[k].field:set_value(1)
        end

        if v > 1 then
            inputMap[k].field:set_value(1)
            inputStates[k] = inputStates[k] - 1
        end
    end
end

function startHold(button)
    inputMap[button].field:set_value(1)
    inputStates[button] = -1
end

function stopHold(button)
    inputMap[button].field:set_value(0)
    inputStates[button] = 0
end

function singlePress(button)
    inputMap[button].field:set_value(1)
    inputStates[button] = tickRate + 1
end

function performDirection(state, posKey, negKey)
    if state == '1' then
        stopHold(posKey)
        singlePress(negKey)
    end

    if state == '2' then
        stopHold(negKey)
        singlePress(posKey)
    end

    if state == '0' then
        stopHold(posKey)
        stopHold(negKey)
    end
end

function performButton(state, key)
    if state == '1' then
        singlePress(key)
    end

    if state == '0' then
        stopHold(key)
    end
end

function performAction(action)
    local vertical = action:sub(1, 1)
    performDirection(vertical, 'U', 'D')
    local horizontal = action:sub(2, 2)
    performDirection(horizontal, 'R', 'L')

    local button1 = action:sub(3, 3)
    performButton(button1, '1')
    local button2 = action:sub(4, 4)
    performButton(button2, '2')
end

function render(screen)
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

exports.inputs = inputMap
exports.states = buttonStates

exports.init = init 
exports.readButtonStates = readButtonStates
exports.updateInputStates = updateInputStates
exports.startHold = startHold
exports.stopHold = stopHold
exports.singlePress = singlePress
exports.render = render
exports.performAction = performAction

return exports
