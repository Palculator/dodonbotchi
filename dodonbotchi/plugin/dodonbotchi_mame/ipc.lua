local json = require('json')

local socket = nil

function sendMessage(message)
  socket:write(message .. '\n')
end

function sendACK()
  local message = {message = 'ACK'}
  message = json.stringify(message)
  sendMessage(message)
end

function readMessage()
  local message = socket:read(4096)
  if message ~= nil and #message > 0 then
    message = json.parse(message)
    return message
  end
  
  return nil
end

function init()
  socket = emu.file('rw')
  socket:open('socket.{{host}}:{{port}}')
end

exports = {}

exports.sendMessage = sendMessage
exports.sendACK = sendACK

exports.readMessage = readMessage

exports.init = init

return exports
