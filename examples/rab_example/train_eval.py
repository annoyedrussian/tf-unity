import os
import gin

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from tf_agents.utils import common
from tf_agents.eval import metric_utils
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

flags.DEFINE_string('root_dir', os.getcwd(), 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_path', None, 'Path to an executable.')
flags.DEFINE_integer('num_iterations', 50000, 'Total number train/eval iterations to perform.')
flags.DEFINE_string('gin_file', None, 'Paths to the gin-config file.')

FLAGS = flags.FLAGS

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


@gin.configurable
def train_eval(
        root_dir,
        env_path,
        discount=0.9,
        num_iterations=50000,
        num_pretrain_iterations=50000,
        eval_interval=5000,
        num_eval_episodes=1,
        collect_interval=1000,
        log_interval=500,
        fc_layer_params=(256, 128),
        batch_size=64,
        learning_rate=1e-3,
        collect_steps_per_iteration=500,
        initial_collect=False,
        initial_collect_steps=1000,
        replay_buffer_capacity=100000,
        summary_interval=1000,
        summaries_flush_secs=10):
    """A train and eval for DQN with unity environment"""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()

    with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(global_step % summary_interval, 0)):
        py_env = suite_unity.load(
        env_path,
        discount=discount,
        worker_id=1,
        use_visual=False)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

        py_eval_env = suite_unity.load(
            env_path,
            discount=discount,
            worker_id=2,
            use_visual=False)
        tf_eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)

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

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        tf_agent.collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=1,
    )
    collect_step_driver = dynamic_step_driver.DynamicStepDriver(
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
    eval_step_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        tf_agent.policy,
        observers=[eval_avg_return_metric],
        num_steps=5000)

        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))

        train_checkpointer.initialize_or_restore()

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    rb_iterator = iter(dataset)

        def eval_step():
            metric_utils.eager_compute(
                eval_metrics,
                tf_eval_env,
                tf_agent.policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics')
            metric_utils.log_metrics(eval_metrics)

    def train_step():
        experience, _ = next(rb_iterator)
        return tf_agent.train(experience)

    time_step = None
    policy_state = tf_agent.collect_policy.get_initial_state(tf_env.batch_size)

    traj_data = decode_protobuf('protobuf/example3.b64', transform_protobuf)
    populate_replay_buffer(traj_data, replay_buffer, discount=discount)

        # eval_step()

    for _ in range(num_pretrain_iterations):
        train_step()
        step = tf_agent.train_step_counter.numpy()
        if step % log_interval == 0:
            print('step {}'.format(step))

        eval_step()

    for _ in range(num_iterations):
        train_step()

        step = tf_agent.train_step_counter.numpy()

        if step % collect_interval == 0:
            tf_env.reset()
            collect_driver.run()

        if step % log_interval == 0:
            print('step {}'.format(step))

        if step % eval_interval == 0:
                train_checkpointer.save(global_step=global_step.numpy())
                eval_step()

def main(_):
    """Main"""
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    if FLAGS.gin_file is not None:
        gin_file_path = os.path.join(FLAGS.root_dir, FLAGS.gin_file)
        gin.parse_config_file(gin_file_path)
    train_eval(FLAGS.root_dir, env_path=FLAGS.env_path, num_iterations=FLAGS.num_iterations)

if __name__ == '__main__':
    flags.mark_flag_as_required('env_path')
    app.run(main)
