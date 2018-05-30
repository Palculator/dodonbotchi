local exports = {}

local COLOUR_SHIP_FILL = 0xAAFF00FF
local COLOUR_SHIP_LINE = 0xAAFF0000

local COLOUR_ENEMY_FILL = 0xAA0000FF
local COLOUR_ENEMY_LINE = 0xAAFFFF00

local COLOUR_BULLET_FILL = 0xAAFF0000
local COLOUR_BULLET_LINE = 0xAA00FFFF

local COLOUR_OWNSHOT_FILL = 0xAA00FFFF
local COLOUR_OWNSHOT_LINE = 0xAAFF0000

local COLOUR_BONUS_FILL = 0xAA8800FF
local COLOUR_BONUS_LINE = 0xFFFF0088

local COLOUR_POWERUP_FILL = 0xAA00FF00
local COLOUR_POWERUP_LINE = 0xFFFF00FF


local ENEMIES_BEG = 0x104AF6
local ENEMIES_END = 0x106836

local BULLETS_BEG = 0x106CB6
local BULLETS_END = 0x108CB6

local POWERUP_BEG = 0x10A9F6
local POWERUP_END = 0x10AB16

local BONUSES_BEG = 0x10AB36
local BONUSES_END = 0x10BDF6

local OWNSHOT_BEG = 0x102D8E
local OWNSHOT_END = 0x1038F6

local SHIP_X = 0x102C92
local SHIP_Y = 0x102C94

local LIVES = 0x101965
local BOMBS = 0x102CB0
local SCORE = 0x10161E
local COMBO = 0x1017D1
local HIT = 0x1017EA

local mem = nil
local screen = nil

local lastState = nil

local screenMaxX = 320
local screenMaxY = 240

local sprt = nil

local function readObjects(ret, addr, addr_end, step)
    if step == nil  then
        step = 0x20
    end

    for i = addr, addr_end, step do
        local id = mem:read_u16(i + 0)
        if id ~= 0 then
            local sid = mem:read_u32(i + 2)
            local pos_x = mem:read_u16(i + 6)
            local pos_y = mem:read_u16(i + 8)
            pos_x = math.floor(pos_x / 64)
            pos_y = math.floor(pos_y / 64)
            if pos_x < screenMaxX and pos_y < screenMaxY then
                local mode = mem:read_u16(i + 10)
                local siz_x = 16 * math.floor(mode / 256)
                local siz_y = 16 * (mode % 256)

                local obj = {id = id,
                             sid = sid, 
                             pos_x = pos_x,
                             pos_y = pos_y,
                             siz_x = siz_x,
                             siz_y = siz_y,
                             mode = mode}
                table.insert(ret, obj)
            end
        end
    end
end

local function filterInvisible(objects, visible)
    local ret = {}
    for i, obj in pairs(objects) do
        local sid = obj.sid
        if visible[sid] then
            table.insert(ret, obj)
        end
    end
    return ret
end

local function readShip()
    local x = mem:read_u16(SHIP_X)
    local y = mem:read_u16(SHIP_Y)

    x = math.floor(x / 64)
    y = math.floor(y / 64)

    return {x = x, y = y}
end

local function weirdToInt(weird, digits)
    local score = weird % 16
    local power = 10
    for i = 0, digits - 1 do
        weird = math.floor(weird / 16)
        score = score + (weird % 16) * power
        power = power * 10
    end
    return score
end

local function readScore()
    -- The score is in a weird format where each digit of the hex-encoded
    -- number is the digit of the score in its decimal form. 0x11814 = 11814
    -- Final number is multiplied by 10 because the last digit of the score is
    -- the credit count.
    local weird = mem:read_u32(SCORE)
    local score = weirdToInt(weird, 7)
    return score * 10
end

local function readLives()
    local lives = mem:read_u8(LIVES)
    return lives
end

local function readBombs()
    local bombs = mem:read_u8(BOMBS)
    return bombs
end

local function readCombo()
    local combo = mem:read_u8(COMBO)
    return combo
end

local function readHit()
    local weird = mem:read_u16(HIT)
    local hit = weirdToInt(weird, 4)
    return hit
end

local function readGameState()
    local layer1 = sprt.readSprites(mem, 0)
    local layer2 = sprt.readSprites(mem, 1)
    local visible = {}
    for i = 1, #layer1 do
        local sid = layer1[i].sid
        visible[sid] = true
    end
    for i = 1, #layer2 do
        local sid = layer2[i].sid
        visible[sid] = true
    end

    local frame = screen:frame_number()

    local ship = readShip()

    local enemies = {}
    local bullets = {}
    local ownshot = {}
    local bonuses = {}
    local powerup = {}

    readObjects(enemies, ENEMIES_BEG, ENEMIES_END)
    readObjects(bullets, BULLETS_BEG, BULLETS_END, 0x40)
    readObjects(bonuses, BONUSES_BEG, BONUSES_END)
    readObjects(powerup, POWERUP_BEG, POWERUP_END)

    readObjects(ownshot, OWNSHOT_BEG, OWNSHOT_END, 0x28)

    enemies = filterInvisible(enemies, visible)

    local lives = readLives()
    local bombs = readBombs()
    local score = readScore()
    local combo = readCombo()
    local hit = readHit()

    local state = {
        frame = frame,
        ship = ship,

        enemies = enemies,
        bullets = bullets,
        ownshot = ownshot,
        bonuses = bonuses,
        powerup = powerup,

        lives = lives,
        bombs = bombs,
        score = score,
        combo = combo,
        hit = hit
    }

    lastState = state

    return state
end

local function init(sprite)
    mem = manager:machine().devices[':maincpu'].spaces['program']
    screen = manager:machine().screens[':screen']
    sprt = sprite
end

local function render(state)
    if state == nil then
        state = lastState
        if state == nil then
            state = readGameState()
        end
    end

    local ship = state.ship

    local enemies = state.enemies
    local bullets = state.bullets
    local ownshot = state.ownshot
    local bonuses = state.bonuses
    local powerup = state.powerup

    screen:draw_box(ship.x - 3, ship.y - 3, ship.x + 3, ship.y + 3, COLOUR_SHIP_FILL, COLOUR_SHIP_LINE)
    screen:draw_line(0, screenMaxY / 2, ship.x, ship.y, COLOUR_SHIP_LINE)

    for i, v in pairs(enemies) do
        screen:draw_box(v.pos_x - 3, v.pos_y - 3, v.pos_x + 3, v.pos_y + 3, COLOUR_ENEMY_FILL, COLOUR_ENEMY_LINE)

        local split = math.floor((v.pos_x - ship.x) * 0.75) + ship.x

        screen:draw_line(ship.x, ship.y, split, v.pos_y, COLOUR_ENEMY_LINE)
        screen:draw_line(split, v.pos_y, v.pos_x, v.pos_y, COLOUR_ENEMY_LINE)
    end

    for i, v in pairs(bullets) do
        screen:draw_box(v.pos_x - 3, v.pos_y - 3, v.pos_x + 3, v.pos_y + 3, COLOUR_BULLET_FILL, COLOUR_BULLET_LINE)
        screen:draw_line(ship.x, ship.y, v.pos_x, v.pos_y, COLOUR_BULLET_LINE)
    end

    for i, v in pairs(ownshot) do
        screen:draw_box(v.pos_x - 3, v.pos_y - 3, v.pos_x + 3, v.pos_y + 3, COLOUR_OWNSHOT_FILL, COLOUR_OWNSHOT_LINE)
    end

    for i, v in pairs(bonuses) do
        screen:draw_box(v.pos_x - 4, v.pos_y - 4, v.pos_x + 4, v.pos_y + 4, COLOUR_BONUS_FILL, COLOUR_BONUS_LINE)
    end

    for i, v in pairs(powerup) do
        screen:draw_box(v.pos_x - 5, v.pos_y - 5, v.pos_x + 5, v.pos_y + 5, COLOUR_POWERUP_FILL, COLOUR_POWERUP_LINE)
    end
end

exports.init = init
exports.readGameState = readGameState
exports.render = render

return exports
