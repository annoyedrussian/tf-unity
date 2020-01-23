from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step

class PlayerPolicy(tf_policy.Base):
    def __init__(self, time_step_spec, action_spec, name=None):
        super(PlayerPolicy, self).__init__(time_step_spec, action_spec)

        self._last_action = None
        self._action_max_repeat = 10
        self._action_repeat_counter = 0

        self._policy_info = ()
    
    def _action(self, time_step, policy_state, seed):
        if self._last_action is None:
            self._action_repeat_counter = 0
            try:
                # print('Take action from {}'.format(self.action_spec))
                action = int(input())
                # print('Your action is {}'.format(action))
            except:
                action = 0
                pass
            self._last_action = action
        else:
            action = self._last_action
            self._action_repeat_counter += 1
            if self._action_repeat_counter == self._action_max_repeat:
                self._last_action = None
        return policy_step.PolicyStep(action, policy_state, self._policy_info)
