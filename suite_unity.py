from gym_unity.envs import UnityEnv
from tf_agents.environments import suite_gym

def load(env_path, **kwargs):
	"""Loads the selected unity environment from path provided.

	Args:
		env_path: A path to Unity executable

	Returns:
		A PyEnvironment instance.
	"""

	unity_env = UnityEnv(env_path, **kwargs)

	return wrap_env(unity_env)

def wrap_env(unity_env):
	"""Wraps given gym environment with TF Agent's GymWrapper.

	Args:
		unity_env: An instance of Unity environment

	Returns:
		A PyEnvironment instance.
	"""
	return suite_gym.wrap_env(unity_env)