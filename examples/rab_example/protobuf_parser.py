import base64

import numpy as np

from tf_agents.trajectories import time_step as ts

from protobuf.message_pb2 import ProtoGameDetail

DEBUG = True

def log(msg):
    if DEBUG:
        print(msg)

def decode_protobuf(filename, prepare_fn=None):
    with open(filename) as protofile:
        msg_bytes = base64.b64decode(protofile.read())
        protobuf = ProtoGameDetail()
        protobuf.ParseFromString(msg_bytes)

        if prepare_fn:
            return prepare_fn(protobuf)
        return protobuf

def transform_protobuf(protobuf):
    """ Returns numpy corresponding to tf agents time_step properties"""

    log('transform_protobuf started')

    log('ClawOpen length is {}'.format(len(protobuf.ClawOpen)))
    log('BlockCount length is {}'.format(len(protobuf.BlockCount)))
    log('BlocksPosition length is {}'.format(len(protobuf.BlocksPosition)))
    log('CurrentBPScore length is {}'.format(len(protobuf.CurrentBPScore)))

    protobuf_len = len(protobuf.CurrentBPScore)
    max_blocks = protobuf.BlockCount[0]

    log('protobuf contains {} steps'.format(protobuf_len))
    log('max blocks value is {}'.format(max_blocks))

    log('block positions recieved {}'.format(len(protobuf.BlocksPosition)))
    observations = [
        np.expand_dims(np.array(protobuf.BlockCount, dtype=np.int32), axis=1),
        np.array(protobuf.BlocksPosition, dtype=np.float).reshape(protobuf_len, max_blocks * 2),
        np.expand_dims(np.array(protobuf.ClawCartPosition, dtype=np.float), axis=1),
        # missing claw pos x and y
        np.array(protobuf.rotationGlobalClawArm, dtype=np.float).reshape(protobuf_len, 3),
        np.array(protobuf.rotationClawArm, dtype=np.float).reshape(protobuf_len, 3),
        np.expand_dims(np.array(protobuf.ClawCartVelocity, dtype=np.float), axis=1),
        # # missing claw ground
        # # missing claw closed
        np.expand_dims(np.array(protobuf.CurrentBPScore, dtype=np.int32), axis=1),
        np.expand_dims(np.array(protobuf.CurrentScore, dtype=np.int32), axis=1),
    ]
    observations = np.hstack(observations)

    actions = np.reshape(
        np.array(protobuf.PlayerControls, dtype=np.int32),
        (protobuf_len, 10))

    def transform_rewards(score):
        nonlocal prev_score
        reward = score - prev_score
        prev_score = score
        return reward

    def get_step_types(score):
        nonlocal prev_score
        nonlocal prev_step_type

        if prev_step_type == ts.StepType.LAST or prev_step_type is None:
            return ts.StepType.FIRST
        step_type = ts.StepType.LAST if score > prev_score else ts.StepType.MID
        prev_score = score
        return step_type

    prev_score = 0
    rewards = np.array(
        list(map(transform_rewards, protobuf.CurrentScore)),
        dtype=np.int32)
    prev_score = 0
    prev_step_type = None
    step_types = np.array(
        list(map(get_step_types, protobuf.CurrentScore)),
        dtype=np.int32)
    log('step types len is {}'.format(len(step_types)))
    log('next step types len is {}'.format(len(next_step_types)))

    return {
        'actions': actions,
        'observations': observations,
        'rewards': rewards,
        'step_types': step_types,
        'next_step_type': next_step_type,
    }

print(decode_protobuf('protobuf/example1.b64', transform_protobuf)['step_types'])
