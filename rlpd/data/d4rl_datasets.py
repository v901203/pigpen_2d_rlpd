#import d4rl
#import gym
import numpy as np

from rlpd.data.dataset import Dataset


class D4RLDataset(Dataset):
    def __init__(self, dataset, clip_to_eps: bool = True, eps: float = 1e-5):
        """
        dataset: list of episodes, each episode is a dict with keys:
            - observations
            - next_observations
            - actions
            - rewards
            - terminals
        """

        obs_list, next_obs_list, act_list, rew_list, masks_list, done_list = [], [], [], [], [], []

        for ep in dataset:
            obs = ep["observations"].astype(np.float32)
            next_obs = ep["next_observations"].astype(np.float32)
            act = ep["actions"].astype(np.float32)
            rew = ep["rewards"].astype(np.float32)
            masks = np.ones(rew.shape)
            done = ep["terminals"].astype(bool)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            act_list.append(act)
            rew_list.append(rew)
            masks_list.append(masks)
            done_list.append(done)

        # Flatten everything into one big array
        self.observations = np.concatenate(obs_list, axis=0)
        self.next_observations = np.concatenate(next_obs_list, axis=0)
        self.actions = np.concatenate(act_list, axis=0)
        self.rewards = np.concatenate(rew_list, axis=0)
        self.masks = np.concatenate(masks_list, axis=0)
        self.dones = np.concatenate(done_list, axis=0)

        if clip_to_eps:
            lim = 1 - eps
            self.actions = np.clip(self.actions, -lim, lim)

        print(f"\033[32mDataset built with {len(self.observations)} transitions.\033[0m")
        
        dataset_dict = {
            'observations': self.observations,
            'next_observations': self.next_observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'masks': self.masks,
            'dones': self.dones,
            }

        '''
        #dataset_dict = d4rl.qlearning_dataset(env)
        
        #N = dataset['rewards'].shape[0]
        #dataset_dict = self.pad_episodes(dataset)
        N = len(dataset)
        print('\033[32m' + f"The dataset has {N} episodes." + '\033[0m')

        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        terminate_on_end=False
        max_episode_steps = 100 #temporary
        for i in range(N):
            obs = dataset[i]['observations'].astype(np.float32)
            new_obs = dataset[i]['next_observations'].astype(np.float32)
            action = dataset[i]['actions'].astype(np.float32)
            reward = dataset[i]['rewards'].astype(np.float32)
            done_bool = dataset[i]['terminals']

            if use_timeouts:
                final_timestep = dataset[i]['timeouts']
            else:
                final_timestep = (episode_step == max_episode_steps - 1)
            if (not terminate_on_end) and final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                episode_step = 0
                continue
            if done_bool or final_timestep:
                episode_step = 0

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1


        for i in range(len(obs_)):
            print(f"obs_[{i}]:{obs_[i]}, obs_[i].dtype:{obs_[i].dtype}, obs_[i].shape:{obs_[i].shape}")
        '''
        
        super().__init__(dataset_dict)

    
    def pad_episodes(self, episodes):
        # episodes is your list-of-dicts dataset

        # Find max length among all episodes
        max_len = max(ep["observations"].shape[0] for ep in episodes)
        obs_dim = episodes[0]["observations"].shape[1]
        print(f"max_len:{max_len}, act_dim:{episodes[0]['actions'].shape}")
        act_dim = episodes[0]["actions"].shape[1]

        n_eps = len(episodes)

        # Allocate padded arrays
        obs_padded = np.zeros((n_eps, max_len, obs_dim), dtype=np.float32)
        next_obs_padded = np.zeros((n_eps, max_len, obs_dim), dtype=np.float32)
        actions_padded = np.zeros((n_eps, max_len, act_dim), dtype=np.float32)
        rewards_padded = np.zeros((n_eps, max_len), dtype=np.float32)
        terminals_padded = np.zeros((n_eps, max_len), dtype=np.bool_)
        mask = np.zeros((n_eps, max_len), dtype=np.bool_)   # valid steps mask

        # Fill arrays episode by episode
        for i, ep in enumerate(episodes):
            T = ep["observations"].shape[0]

            obs_padded[i, :T] = ep["observations"]
            next_obs_padded[i, :T] = ep["next_observations"]
            actions_padded[i, :T] = ep["actions"]
            rewards_padded[i, :T] = ep["rewards"]
            terminals_padded[i, :T] = ep["terminals"]

            mask[i, :T] = True   # mark valid steps

        return {
            "observations": obs_padded,
            "next_observations": next_obs_padded,
            "actions": actions_padded,
            "rewards": rewards_padded,
            "terminals": terminals_padded,
        }
