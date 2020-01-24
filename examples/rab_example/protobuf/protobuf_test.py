import base64

from message_pb2 import ProtoGameDetail

with open("example1.b64") as file:
    msg_file = file.read()
    msg_bytes = base64.b64decode(msg_file)

msg = ProtoGameDetail()
msg.ParseFromString(msg_bytes)

print('Msg debug info')
print(msg.Name)
print(msg.FeedBack)
asd = len(msg.BlockCount)
print(len(msg.BlockCount))
print(len(msg.CurrentScore))
print(len(msg.PlayerControls) / asd)

asd = [x for x in msg.ClawController if x > 0]
print(len(asd))