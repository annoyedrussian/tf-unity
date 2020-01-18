# import gym
from gym_unity.envs import UnityEnv

env = UnityEnv("./grid-world-macos-3.app", worker_id=1, use_visual=True, uint8_visual=True)
print(env.action_space)

import tensorflow as tf

from tf_agents.environments import suite_gym

from tf_agents.environments import utils

from tf_agents.networks import q_network
from tf_agents.metrics import tf_metrics
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.policies import random_py_policy
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.utils import common

tf_py_env = suite_gym.wrap_env(env, discount=0.7)
# validate won't work
# see random_walk function below
# utils.validate_py_environment(tf_py_env)

def tf_action_to_unity_action(action):
    return [[action.numpy()[0]]]

def random_walk(env, num_episodes):
    random_policy = random_py_policy.RandomPyPolicy(
        env.time_step_spec(),
        env.action_spec()
    )
    time_step = env.reset()
    for _ in range(num_episodes):
        action_step = random_policy.action(time_step)
        time_step = env.step([[action_step.action]])
        # print(time_step.reward)

# random_walk(tf_py_env, 10000000)

tf_env = tf_py_environment.TFPyEnvironment(tf_py_env)

fc_layer_params = (512, 256, 128)
q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=fc_layer_params,
    # conv_layer_params=(64, (3, 3), 1)
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss
)
agent.initialize()

def compute_avg_return(
    environment,
    policy,
    num_episodes=10
):
  total_return = 0.0
  total_length = 0

  for _ in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0
    episode_length = 0
    while not time_step.is_last():
      action_step = policy.action(time_step)
      action = action_step.action.numpy()[0]
      time_step = environment.step([[action]])
      episode_return += time_step.reward
      episode_length += 1
    total_return += episode_return
    total_length += episode_length

  avg_return = total_return / num_episodes
  avg_length = total_length / num_episodes
  return avg_return.numpy()[0], avg_length


random_tf_policy = random_tf_policy.RandomTFPolicy(
    tf_env.time_step_spec(),
    tf_env.action_spec()
)
# print(compute_avg_return(tf_env, random_tf_policy))

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(tf_action_to_unity_action(action_step.action))
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)
  return next_time_step

def collect_episode(environment, policy, buffer):
    time_step = environment.reset()
    while not time_step.is_last():
        time_step = collect_step(environment, policy, buffer)

def collect_data_episodes(env, policy, buffer, episodes=1):
    for _ in range(episodes):
        collect_episode(env, policy, buffer)

def collect_data_steps(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

def play(tf_env):
    done = False
    tf_env.reset()
    while not done:
        action = int(input('select action [0, 4] - '))
        time_step = tf_env.step([[action]])
        print('reward is %d' % time_step.reward.numpy())

num_iterations = 10000
eval_interval = num_iterations / 100
collect_episodes_per_iteration = 10
collect_steps_per_iteration = 100
replay_buffer_max_length = 10000

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_max_length)
replay_buffer_dataset = replay_buffer.as_dataset(
    num_steps=2,
    sample_batch_size=64,
).prefetch(3)
replay_buffer_iterator = iter(replay_buffer_dataset)

# collect_data_episodes(tf_env, agent.collect_policy, replay_buffer, 100)
# print(compute_avg_return(tf_env, agent.policy))

print('statring training loop for %d iterations' % num_iterations)
for iteration in range(num_iterations):
    # collect_data_step(tf_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
    collect_data_episodes(tf_env, agent.collect_policy, replay_buffer, collect_episodes_per_iteration)

    experience, _ = next(replay_buffer_iterator)
    agent.train(experience)

    if iteration % eval_interval == 0:
        avg_return, avg_length = compute_avg_return(tf_env, agent.policy)
        print("%d/%d reward: %f length: %d" % (iteration, num_iterations, avg_return, avg_length))
