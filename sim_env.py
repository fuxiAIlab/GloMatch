import os
import os.path as osp
import time
import math
import itertools
import numpy as np
import pandas as pd
import torch

import gym
from gym import spaces, logger
from gym.utils import seeding

from glomatch.config import data_dir, model_dir, MODEL_CONFIG
from glomatch.model import LRModel
from glomatch.utils import preprocess_profile


def load_data(conf):
    file_name = "profile-demo.csv"
    # csv_file = osp.join(data_dir, file_name)
    csv_file = os.path.join(data_dir, file_name)
    print("load data:", csv_file)
    df = pd.read_csv(csv_file, sep=',', header='infer')
    # df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')
    data = preprocess_profile(df)
    # data = np.random.uniform(low=0.0, high=1.0, size=(10000, conf['num_features']))
    print(data.shape, type(data))
    # https://stackoverflow.com/questions/911871/detect-if-a-numpy-array-contains-at-least-one-non-numeric-value
    print("Test element-wise for NaN: ", np.isnan(data).any())
    return data


def load_model(conf):
    model = LRModel(conf).to(conf['device'])
    model_name = type(model).__name__
    file_name = '%s-%s-%s-20.pt' % (conf['dataset'], conf['label'], model_name)
    path = os.path.join(model_dir, file_name)
    print("load model:", path)
    # If you are running on a CPU-only machine,
    # please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


class MMEnv(gym.Env):
    """
    Description:
        Matchmaking Environment.

    Observation:
        Type: Box()
        Num	Observation    Min  Max
        0	Player Feature 1  -Inf  Inf
        1	Player Feature 2  -Inf  Inf
        ...

    Actions:
        Type: Discrete(N)
        Num	Action
        0	Choose Player 0
        1	Choose Player 1
        ...

    Reward:
        Reward is evaluate when two competing teams are formed.

    Starting State:
        All candidate players in the player pool.

    Episode Termination:
        All players are matched.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
    }

    num_players = 120  # number of candidate players
    team_size = 3  # number of players per team
    num_features = 1  # number of player features

    def __init__(self, conf):

        self.num_players = conf['num_players']
        self.team_size = conf['team_size']
        self.num_features = conf['num_features']

        self.x = None  # feature array for player pool (num_players, num_features)
        self.sort = True  # whether sort players by elo or not

        self.data = load_data(conf)  # feature array for all players
        self.all_players = len(self.data)  # number of all players
        self.model = load_model(conf)  # model object

        high = 9999
        self.action_space = spaces.Discrete(self.num_players)
        # self.observation_space = spaces.Box(-1.0, 1.0, shape=(2 * self.num_players * self.num_features,))
        self.observation_space = spaces.Dict({
            "mask": spaces.Box(0, 1, shape=(self.num_players,)),
            "team": spaces.Box(-high, high, shape=(2 * self.team_size, self.num_features)),
            "pool": spaces.Box(-high, high, shape=(self.num_players, self.num_features)),
            "state": spaces.Box(-high, high, shape=(2 * self.team_size + self.num_players, self.num_features)),
        })
        self.reward_range = (-float('inf'), float('inf'))

        self.state = None
        self.seed()

        self.max_episode_steps = self.num_players
        self.elapsed_steps = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reward(self, team, team_draft):
        assert len(team_draft) == 2 * self.team_size
        assert self.elapsed_steps % (2 * self.team_size) == 0
        x = team.reshape(-1, self.num_features)
        idx = np.arange(2 * self.team_size)
        idx1 = (idx % 2 == 0)
        idx2 = np.logical_not(idx1)
        x1 = x[idx1][np.newaxis, :]  # (1, T, F)
        x2 = x[idx2][np.newaxis, :]  # (1, T, F)
        outcome = self.model.predict((x1, x2))  # (1, 1)
        # print(outcome.shape, outcome)
        # x.ravel() / x.reshape(-1) / x.squeeze(axis=1) / x.flatten()
        return -np.abs(np.ravel(outcome) - 0.5)[0]

    def step(self, action):
        assert self.elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        assert action in self.action_space, "%r (%s) invalid" % (action, type(action))
        self.elapsed_steps += 1
        n_chosed = self.elapsed_steps % (2 * self.team_size)
        team_done = bool(n_chosed == 0)

        team_draft, pool_mask, team, pool = self.state

        if team_done:  # n_chosed is 0
            team_draft[-1] = action
            team[-1] = self.x[action]
            start_time = time.time()
            reward = self._reward(team, team_draft)
            elapsed = time.time() - start_time
            # print("team composition: ", self.elapsed_steps, team_draft, team, reward)
            team_draft[:] = -1
            team[:] = 0
        else:  # 1 <= n_chosed <= team_size - 1
            team_draft[n_chosed - 1] = action
            team[n_chosed - 1] = self.x[action]
            reward = 0.0
            elapsed = 0.0

        pool_mask[action] = 0
        pool[action] = 0.0

        self.state = (team_draft, pool_mask, team, pool)

        done = True if self.elapsed_steps >= self.max_episode_steps else False
        # return np.array(self.state), reward, done, {}
        # return np.array(np.concatenate(self.state)), reward, done, {}
        # return list(map(np.array, self.state)), reward, done, {}
        return {
            "mask": np.array(pool_mask),
            "team": np.array(team),
            "pool": np.array(pool),
            "state": np.concatenate((team, pool))
        }, reward, done, {"elapsed": elapsed}

    def reset(self):
        players = np.random.choice(self.all_players, self.num_players, replace=False)
        x = [self.data[p] for p in players]
        x = np.array(x)
        if self.sort:
            y = x[:, 0]  # ability_score
            indices = np.argsort(y)
            self.x = x[indices]
        else:
            self.x = x
        team_draft = np.array([-1] * 2 * self.team_size, dtype=np.int32)
        pool_mask = np.array([1] * self.num_players, dtype=np.int32)
        team = np.zeros(shape=(2 * self.team_size, self.num_features))
        pool = self.x[np.arange(self.num_players)]
        self.state = (team_draft, pool_mask, team, pool)
        self.elapsed_steps = 0
        # return np.array(self.state)
        # return np.array(np.concatenate(self.state))
        # return list(map(np.array, self.state))
        return {
            "mask": np.array(pool_mask),
            "team": np.array(team),
            "pool": np.array(pool),
            "state": np.concatenate((team, pool))
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':

    conf = dict()
    conf['dataset'] = 'ball'
    conf['label'] = 'win'
    conf['algo'] = 'lr'
    conf['device'] = 'cuda'
    conf['num_players'] = 120
    conf['team_size'] = 3
    conf['num_features'] = 19
    conf['num_classes'] = 1

    conf.update(MODEL_CONFIG[conf['algo']])

    env = MMEnv(conf)
    state = env.reset()
    print("Initial State: \n", state)

    num_players = env.num_players
    actions = np.random.permutation(range(0, num_players))
    # actions = list(itertools.permutations(range(1, num_players + 1), 1))
    print("Test actions: \n", actions)

    for i in range(num_players):
        env.render()
        # action = np.random.randint(low=1, high=8 + 1)  # this takes random actions
        action = actions[i]
        next_state, reward, done, info = env.step(action)
        print('-' * 80)
        print("Env step %d:" % (i + 1), action, reward, done, info)
        # print("Env step %d:" % (i + 1), state, action, next_state, reward, done)
        # print("Env step %d:" % (i + 1), state, action, next_state, reward, done, sep='\n', end='\n')
        state = next_state
        if done:
            state = env.reset()
    env.close()
