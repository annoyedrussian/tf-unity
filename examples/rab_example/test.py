from gym_unity.envs import UnityEnv

unity_env = UnityEnv(
    'executables/RABEnvironment_0_001.app',
    worker_id=1,
    use_visual=False,
    allow_multiple_visual_obs=True)

time_step = unity_env.step([1])
print(len(time_step[0]))
print(time_step[0])