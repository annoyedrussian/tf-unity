import os

import tensorflow as tf

from tf_agents.utils import common
from tf_agents.networks import q_network
from tf_agents.metrics import tf_metrics
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from protobuf_parser import decode_protobuf, transform_protobuf

import suite_unity

def populate_replay_buffer(traj_data, replay_buffer, discount=1.0):
    """ Populates replay buffer with decoded trajectory data

    Args:
        traj_data: Decoded trajectory data
        replay_buffer: Replay buffer
        discount: Environment dicsount

    Returns:
        None
    """
    length = len(traj_data['step_types'])

    for idx in range(length):
        step_type = tf.convert_to_tensor(
            traj_data['step_types'][idx], traj_data['step_types'].dtype)
        observation = tf.convert_to_tensor(
            traj_data['observations'][idx], traj_data['observations'].dtype)
        action = tf.convert_to_tensor(traj_data['actions'][idx], traj_data['actions'].dtype)
        policy_info = ()
        next_step_type = tf.convert_to_tensor(
            traj_data['next_step_types'][idx], traj_data['next_step_types'].dtype)
        reward = tf.convert_to_tensor(traj_data['rewards'][idx], traj_data['rewards'].dtype)
        discount = tf.convert_to_tensor(discount, tf.float32)
        traj = trajectory.Trajectory(
            step_type, observation, action, policy_info, next_step_type, reward, discount)
        replay_buffer.add_batch(traj)

# todo: add checkpoints
# todo: add tensorboard and summaries

def train_eval(
        root_dir,
        env_name='RABMLPiscine3.0_AdjustedTraining.app',
        discount=0.7,
        num_iterations=10000,
        num_pretrain_iterations=1000,
        eval_interval=1000,
        collect_interval=100,
        log_interval=100,
        fc_layer_params=(256, 256, 128),
        batch_size=64,
        learning_rate=1e-3,
        collect_steps_per_iteration=100,
        initial_collect=False,
        initial_collect_steps=1000,
        replay_buffer_capacity=100000):
    """A train and eval for DQN with unity environment"""

    executables_dir = os.path.join(root_dir, 'executables')
    env_path = os.path.join(executables_dir, env_name)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_py_env = suite_unity.load(
        env_path,
        discount=discount,
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
    tf_agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    inital_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())

    if initial_collect:
    dynamic_step_driver.DynamicStepDriver(
        tf_env,
        inital_collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps).run()

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        tf_agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration)

    eval_avg_return_metric = tf_metrics.AverageReturnMetric()
    eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        tf_agent.policy,
        observers=[eval_avg_return_metric],
        num_episodes=1)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    rb_iterator = iter(dataset)

    def train_step():
        experience, _ = next(rb_iterator)
        return tf_agent.train(experience)

    time_step = None
    policy_state = tf_agent.collect_policy.get_initial_state(tf_env.batch_size)

    traj_data = decode_protobuf('protobuf/example3.b64', transform_protobuf)
    populate_replay_buffer(traj_data, replay_buffer, discount=discount)

    for _ in range(num_iterations):
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state
        )
        train_step()

def main():
    """Main"""
    train_eval(os.getcwd())

if __name__ == '__main__':
    main()
