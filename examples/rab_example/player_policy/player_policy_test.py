import os

import suite_unity

from player_policy import PlayerPolicy
from util import display_replay_buffer

import tensorflow as tf

from tf_agents.utils import common
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer

env_path = 'RABMLPiscine.app'
learning_rate = 0.01
fc_layer_params = (120,)
replay_buffer_capacity = 1000
collect_episodes_per_iteration = 1
collect_steps_per_iteration = 500

global_step = tf.compat.v1.train.get_or_create_global_step()

tf_py_env = suite_unity.load(
    env_path,
    discount=0.8,
    worker_id=1,
    use_visual=False)
tf_env = tf_py_environment.TFPyEnvironment(tf_py_env)

q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=fc_layer_params)

tf_agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    td_errors_loss_fn=common.element_wise_squared_loss,
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
    train_step_counter=global_step)

policy = PlayerPolicy(tf_env.time_step_spec(), tf_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_capacity)

rb_checpoint = tf.train.Checkpoint(replay_buffer=replay_buffer)

collect_step_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)
collect_episode_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    tf_env,
    policy,
    observers=[replay_buffer.add_batch],
    num_episodes=collect_episodes_per_iteration)

collect_episode_driver.run()

rb_checpoint.save(file_prefix='replay_buffer/')