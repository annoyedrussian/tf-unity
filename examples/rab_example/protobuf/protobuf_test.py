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