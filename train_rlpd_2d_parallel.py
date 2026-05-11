import os
import datetime
import numpy as np
import gymnasium as gym
from absl import app, flags
from ml_collections import config_flags
from torch.utils.tensorboard import SummaryWriter

try:
    from flax.training import checkpoints
except:
    print('\033[41m' + "Not loading checkpointing functionality." + '\033[0m')

from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd_2d_parallel", "wandb project name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 200, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.") 
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(500000), "Number of training steps.")
flags.DEFINE_integer("start_training", int(2000), "Number of random steps before training starts.")
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_string("expert_data_path", "expert_data.npz", "Path to your recorded expert data.")
flags.DEFINE_integer("num_envs", 4, "Number of parallel environments.")

# 👇 接續訓練專用參數
flags.DEFINE_string("resume_dir", "", "Path to the old checkpoint folder to resume training.")

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def load_expert_data(buffer, filepath):
    if not os.path.exists(filepath):
        print(f'\033[43m找不到專家資料 {filepath}，將進行純粹的線上探索訓練。\033[0m')
        return 0
    print(f'\033[42m載入專家資料 {filepath} 中...\033[0m')
    data = np.load(filepath)
    obs, actions, rewards = data['obs'], data['actions'], data['rewards']
    next_obs, dones = data['next_obs'], data['dones']
    num_samples = len(obs)
    for i in range(num_samples):
        buffer.insert(dict(
            observations=obs[i], actions=actions[i], rewards=rewards[i],
            masks=1.0 if not dones[i] else 0.0, dones=dones[i], next_observations=next_obs[i],
        ))
    print(f'\033[42m成功載入 {num_samples} 筆專家經驗！\033[0m')
    return num_samples

def make_env():
    def _init():
        from pig_pen_env import PigPenEnv
        return PigPenEnv(render_mode=None, door_width_scale=1.0)
    return _init

def main(_):
    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chkpt_dir = os.path.abspath(os.path.join("checkpoints", f"RLPD_2D_Parallel_{run_name}"))
    
    num_envs = FLAGS.num_envs
    print(f'\033[46m啟動平行訓練！同時開啟 {num_envs} 個豬舍環境...\033[0m')
    
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])
    eval_env = make_env()()

    observation_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls", "SACLearner")
    agent = globals()[model_cls].create(FLAGS.seed, observation_dim, action_dim, **kwargs)

    replay_buffer = ReplayBuffer(observation_dim, action_dim, capacity=int(5e5))
    replay_buffer.seed(FLAGS.seed)
    expert_samples = load_expert_data(replay_buffer, FLAGS.expert_data_path)

    total_steps = 0

    # 🌟 接續訓練邏輯 🌟
    if FLAGS.resume_dir:
        abs_resume_dir = os.path.abspath(FLAGS.resume_dir)
        latest_ckpt = checkpoints.latest_checkpoint(abs_resume_dir)
        if latest_ckpt:
            print(f"\033[42m[接續訓練] 找到舊存檔，正在載入大腦: {latest_ckpt}\033[0m")
            agent = checkpoints.restore_checkpoint(latest_ckpt, agent)
            try:
                total_steps = int(os.path.basename(latest_ckpt).replace('checkpoint_', ''))
            except:
                pass
            print(f"\033[42m[接續訓練] 將從第 {total_steps} 步開始繼續修煉！\033[0m")
            # 繼續存在原本的資料夾
            chkpt_dir = abs_resume_dir

    os.makedirs(chkpt_dir, exist_ok=True)
    writer = SummaryWriter(f'runs/RLPD_2D_Parallel_{os.path.basename(chkpt_dir)}')

    obs, info = envs.reset()
    
    while total_steps < FLAGS.max_steps:
        if total_steps < FLAGS.start_training and expert_samples == 0:
            actions = envs.action_space.sample()
        else:
            actions = []
            for i in range(num_envs):
                act, agent = agent.sample_actions(obs[i])
                actions.append(act)
            actions = np.array(actions)

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        
        for i in range(num_envs):
            done = terminated[i] or truncated[i]
            mask = 0.0 if terminated[i] else 1.0 
            actual_next_obs = next_obs[i]
            
            if done and "final_observation" in infos and infos["_final_observation"][i]:
                actual_next_obs = infos["final_observation"][i]

            replay_buffer.insert(dict(
                observations=obs[i], actions=actions[i], rewards=rewards[i],
                masks=mask, dones=done, next_observations=actual_next_obs,
            ))

        obs = next_obs
        total_steps += num_envs

        if total_steps >= FLAGS.start_training or expert_samples > 0:
            for _ in range(num_envs):
                batch = replay_buffer.sample(int(FLAGS.batch_size * FLAGS.utd_ratio))
                agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if total_steps % FLAGS.log_interval < num_envs:
                for k, v in update_info.items():
                    writer.add_scalar(f"training/{k}", float(v), total_steps)

        if total_steps % FLAGS.eval_interval < num_envs:
            print(f'\033[44m[評估] 進行 {total_steps} 步的成效測試...\033[0m')
            avg_reward = 0.
            episodes = 5
            for _ in range(episodes):
                eval_obs, _ = eval_env.reset()
                eval_done = False
                episode_reward = 0
                while not eval_done:
                    eval_action = agent.eval_actions(eval_obs)
                    eval_obs, e_reward, e_term, e_trunc, _ = eval_env.step(np.array(eval_action))
                    eval_done = e_term or e_trunc
                    episode_reward += e_reward
                avg_reward += episode_reward
            avg_reward /= episodes
            
            print(f'\033[44m[評估結果] 平均獎勵: {avg_reward:.2f}\033[0m')
            writer.add_scalar('avg_reward/eval', avg_reward, total_steps)

            if FLAGS.checkpoint_model:
                checkpoints.save_checkpoint(chkpt_dir, agent, step=total_steps, keep=3, overwrite=True)
                print(f'\033[42m模型已儲存至 {chkpt_dir}\033[0m')

    writer.close()
    envs.close()

if __name__ == "__main__":
    app.run(main)