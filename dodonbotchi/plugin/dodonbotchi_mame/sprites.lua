local exports = {}

local layer1Start = 0x400000
local layer1End = 0x404000 - 0x10
local layer2Start = 0x404000
local layer2End = 0x408000 - 0x10

local sprites = {}
local spriteColours = {
    0xFFFF0000,
    0xFF00FF00,
    0xFF0088FF
}

function readSpriteAt(mem, i)
    local sid = mem:read_u64(i + 0x0)
    local pos = sid % 4294967296

    sid = math.floor(sid / 4294967296)

    if sid > 0 then
        local pos_x = math.floor(pos / 65536)
        local pos_y = pos % 65536

        local mode = mem:read_u16(i + 0x8)

        local siz_x = 16 * math.floor(mode / 256)
        local siz_y = 16 * (mode % 256)

        local data = {sid = sid,
                      pos_x = pos_x,
                      pos_y = pos_y,
                      siz_x = siz_x,
                      siz_y = siz_y,
                      mode = mode}
        return data
    else
        return nil
    end
end

function readSprites(mem, layer)
    if layer == nil then
        layer = 0
    end

    local spriteStart, spriteEnd, sid, sid1, sid2, pos, x, y, mode, width, height, data

    if layer == 1 then
        spriteStart = layer1Start
        spriteEnd = layer1End
    else
        spriteStart = layer2Start
        spriteEnd = layer2End
    end

    sprites = {}
    for i = spriteStart, spriteEnd, 0x10 do
        local data = readSpriteAt(mem, i)

        if data ~= nil then
            table.insert(sprites, data)
        end
    end

    return sprites
end

function getSprites()
    return sprites
end

function render(screen)
    local pos, x, y, width, height, colour_idx, colour

    for i, v in pairs(sprites) do
        pos_x = v['pos_x']
        pos_y = v['pos_y']
        siz_x = v['siz_x']
        siz_y = v['siz_y']

        colour_idx = v['sid'] % 3 + 1
        colour = spriteColours[colour_idx]

        screen:draw_box(pos_x, pos_y, pos_x + siz_x, pos_y + siz_y, 0, colour)
        screen:draw_text(pos_x, pos_y + siz_y, string.format('%x', v.sid))
    end
end

exports.readSpriteAt = readSpriteAt
exports.readSprites = readSprites
exports.getSprites = getSprites
exports.render = render

return exports
