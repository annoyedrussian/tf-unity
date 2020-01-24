from gym_unity.envs import UnityEnv

unity_env = UnityEnv(
    'executables/RABMLPiscine3.0_AdjustedTraining.app',
    worker_id=1,
    use_visual=False,
    allow_multiple_visual_obs=True)

time_step = unity_env.step([1])
print(len(time_step[0]))