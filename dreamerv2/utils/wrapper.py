# import minatar
import cv2
import gym
import numpy as np

from gym.spaces.box import Box


class GymAtar(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, display_time=50):
        self.display_time = display_time
        self.env_name = env_name
        self.env = WarpFrame(gym.make(env_name))

        obs_shape = self.env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            self.env = TransposeImage(self.env, op=[2, 0, 1])

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        # self.env.reset()
        # return self.env.state().transpose(2, 0, 1)
        return self.env.reset()

    def step(self, index):
        '''index is the action id, considering only the set of minimal actions'''
        # action = self.minimal_actions[index]
        # r, terminal = self.env.act(action)
        s_, r, terminal, info = self.env.step(index)
        self.game_over = terminal
        # return self.env.state().transpose(2, 0, 1), r, terminal, {}
        return s_, r, terminal, info

    def seed(self, seed='None'):
        # self.env = minatar.Environment(self.env_name, random_seed=seed)
        self.env = gym.make(self.env_name, seed=seed)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            self.env.display_state(self.display_time)

    def close(self):
        if self.env.visualized:
            self.env.close_display()
        return 0


# class breakoutPOMDP(gym.ObservationWrapper):
#     def __init__(self, env):
#         '''index 2 (trail) is removed, which gives ball's direction'''
#         super(breakoutPOMDP, self).__init__(env)
#         c, h, w = env.observation_space.shape
#         self.observation_space = gym.spaces.MultiBinary((c - 1, h, w))
#
#     def observation(self, observation):
#         return np.stack([observation[0], observation[1], observation[3]], axis=0)
#
#
# class asterixPOMDP(gym.ObservationWrapper):
#     '''index 2 (trail) is removed, which gives ball's direction'''
#
#     def __init__(self, env):
#         super(asterixPOMDP, self).__init__(env)
#         c, h, w = env.observation_space.shape
#         self.observation_space = gym.spaces.MultiBinary((c - 1, h, w))
#
#     def observation(self, observation):
#         return np.stack([observation[0], observation[1], observation[3]], axis=0)
#
#
# class freewayPOMDP(gym.ObservationWrapper):
#     '''index 2-6 (trail and speed) are removed, which gives cars' speed and direction'''
#
#     def __init__(self, env):
#         super(freewayPOMDP, self).__init__(env)
#         c, h, w = env.observation_space.shape
#         self.observation_space = gym.spaces.MultiBinary((c - 5, h, w))
#
#     def observation(self, observation):
#         return np.stack([observation[0], observation[1]], axis=0)
#
#
# class space_invadersPOMDP(gym.ObservationWrapper):
#     '''index 2-3 (trail) are removed, which gives aliens' direction'''
#
#     def __init__(self, env):
#         super(space_invadersPOMDP, self).__init__(env)
#         c, h, w = env.observation_space.shape
#         self.observation_space = gym.spaces.MultiBinary((c - 2, h, w))
#
#     def observation(self, observation):
#         return np.stack([observation[0], observation[1], observation[4], observation[5]], axis=0)
#
#
# class seaquestPOMDP(gym.ObservationWrapper):
#     '''index 3 (trail) is removed, which gives enemy and driver's direction'''
#
#     def __init__(self, env):
#         super(seaquestPOMDP, self).__init__(env)
#         c, h, w = env.observation_space.shape
#         self.observation_space = gym.spaces.MultiBinary((c - 1, h, w))
#
#     def observation(self, observation):
#         return np.stack([observation[0], observation[1], observation[2], observation[4], observation[5], observation[6],
#                          observation[7], observation[8], observation[9]], axis=0)


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])
