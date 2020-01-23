import os

import tensorflow as tf

from tf_agents.utils import common
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import suite_unity

# todo: add checkpoints
# todo: add tensorboard and summaries

def train_eval(
        root_dir,
        env_name='RABMLPiscine3.0_AdjustedTraining.app',
        num_iterations=100000,
        fc_layer_params=(100,),
        batch_size=64,
        learning_rate=1e-3,
        collect_steps_per_iteration=1,
        initial_collect_steps=1000,
        replay_buffer_capacity=100000):
    """A train and eval for DQN with unity environment"""

    executables_dir = os.path.join(root_dir, 'executables')
    env_path = os.path.join(executables_dir, env_name)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_py_env = suite_unity.load(
        env_path,
        worker_id=1,
        use_visual=True,
        uint8_visual=True)
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
