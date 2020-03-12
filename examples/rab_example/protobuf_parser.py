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
    log('PlayerControls length is {}'.format(len(protobuf.PlayerControls)))
    log('StepRewards length is {}'.format(len(protobuf.StepReward)))

    protobuf_len = len(protobuf.CurrentBPScore)
    max_blocks = protobuf.BlockCount[0]

    log('protobuf contains {} steps'.format(protobuf_len))
    log('max blocks value is {}'.format(max_blocks))

    log('block positions recieved {}'.format(len(protobuf.BlocksPosition)))

    block_positons = protobuf.BlocksPosition[:protobuf_len * max_blocks * 2]
    claw_positions = protobuf.ClawPosition[:protobuf_len * 2]
    rotation_global_claw_arm = protobuf.rotationGlobalClawArm[:protobuf_len * 3]
    rotation_claw_arm = protobuf.rotationClawArm[:protobuf_len * 3]
    block_count = protobuf.BlockCount[:protobuf_len]
    claw_cart_position = protobuf.ClawPosition[:protobuf_len]
    claw_cart_velocity = protobuf.ClawCartVelocity[:protobuf_len]
    claw_open = protobuf.ClawOpen[:protobuf_len]
    claw_facing_ground = protobuf.ClawFacingGround[:protobuf_len]
    current_bp_score = protobuf.CurrentBPScore[:protobuf_len]

    observations = [
        np.expand_dims(np.array(block_count, dtype=np.float32), axis=1),
        np.array(block_positons, dtype=np.float32).reshape(protobuf_len, max_blocks * 2),
        np.expand_dims(np.array(claw_cart_position, dtype=np.float32), axis=1),
        np.array(claw_positions, dtype=np.float32).reshape(protobuf_len, 2),
        np.array(rotation_global_claw_arm, dtype=np.float32).reshape(protobuf_len, 3),
        np.array(rotation_claw_arm, dtype=np.float32).reshape(protobuf_len, 3),
        np.expand_dims(np.array(claw_cart_velocity, dtype=np.float32), axis=1),
        np.expand_dims(np.array(claw_open, dtype=np.float32), axis=1),
        np.expand_dims(np.array(claw_facing_ground, dtype=np.float32), axis=1),
        np.expand_dims(np.array(current_bp_score, dtype=np.float32), axis=1),
        # np.expand_dims(np.array(protobuf.CurrentScore, dtype=np.int32), axis=1),
    ]
    observations = np.hstack(observations)

    actions = np.reshape(
        np.array(protobuf.PlayerControls, dtype=np.int32),
        (protobuf_len, 9))
    actions = normalize_actions(actions)

    rewards = np.expand_dims(np.array(protobuf.StepReward, dtype=np.float32), axis=1)
    step_types = np.array(get_step_types(protobuf.CurrentScore), dtype=np.int32)
    next_step_types = np.concatenate((step_types[1:], [ts.StepType.LAST]))

    log('step types len is {}'.format(len(step_types)))
    log('next step types len is {}'.format(len(next_step_types)))

    return {
        'actions': actions,
        'observations': observations,
        'rewards': rewards,
        'step_types': step_types,
        'next_step_types': next_step_types,
    }

def get_rewards_from_scores(scores):
    rewards = []

    for score in scores:
        if score != 0:
            rewards.append(score)
        else:
            rewards.append(-1)
    return np.array(rewards, dtype=np.float32)

def normalize_rewards(rewards):
    prev_reward = 0.0
    normalized_rewards = []

    for reward in rewards:
        normalized_rewards.append(reward - prev_reward)
        prev_reward = reward

    return np.array(normalized_rewards, dtype=np.float32)

def get_step_types(scores):
    """
    Returns array of step types
    """
    step_types = [ts.StepType.FIRST]
    prev_score = scores[0]
    for score in scores[1:]:
        curr_step_type = ts.StepType.LAST if score > prev_score else ts.StepType.MID
        prev_score = score
        step_types.append(curr_step_type)
    return step_types

# todo: remove trajs with action 0
def normalize_actions(actions):
    """
    Reduces every action to action index
    Example: action = [0, 0, 0, 0, 1]
    Result: action = [4]
    """
    normalized_actions = []

    for action in actions:
        current_action = np.where(action == 1)[0]
        current_action = current_action[0] + 1 if len(current_action) else 0
        normalized_actions.append([current_action])

    return np.array(normalized_actions, dtype=np.int32)

if __name__ == '__main__':
    asd = decode_protobuf('v1/user_data/ivan2.b64', transform_protobuf)
    # asd = decode_protobuf('v1/user_data/move open claw', transform_protobuf)
    # asd = decode_protobuf('protobuf/ivan.b64', transform_protobuf)