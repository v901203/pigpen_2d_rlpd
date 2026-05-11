import os
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints

# 匯入 RLPD 套件與你的環境
from rlpd.agents import SACLearner
from pig_pen_env import PigPenEnv

FLAGS = flags.FLAGS

# 🌟 改成指定「專案資料夾」即可，不用指定到特定步數
flags.DEFINE_string("checkpoint_dir", "", "Path to the run folder (e.g., checkpoints/RLPD_2D_Parallel_XXX)")
flags.DEFINE_integer("seed", 42, "Random seed.")

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the config used during training.",
    lock_config=False,
)

def main(_):
    if not FLAGS.checkpoint_dir:
        print("\033[41m請在參數中指定 --checkpoint_dir 路徑！(例如: checkpoints/RLPD_2D_Parallel_...)\033[0m")
        return

    # 1. 尋找最新權重
    abs_ckpt_dir = os.path.abspath(FLAGS.checkpoint_dir)
    latest_ckpt = checkpoints.latest_checkpoint(abs_ckpt_dir)
    
    if latest_ckpt is None:
        print(f"\033[41m在 {abs_ckpt_dir} 裡面找不到任何 checkpoint 存檔！\033[0m")
        return

    # 2. 初始化 2D 環境（開畫面）
    env = PigPenEnv(render_mode="human", door_width_scale=1.0)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 3. 建立代理人實體
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls", "SACLearner")
    agent = globals()[model_cls].create(FLAGS.seed, observation_dim, action_dim, **kwargs)

    # 4. 核心：載入最新大腦權重
    print(f"\033[42m自動找到並載入最新權重: {latest_ckpt}\033[0m")
    agent = checkpoints.restore_checkpoint(latest_ckpt, agent)

    # 5. 開始表演
    obs, _ = env.reset()
    print("\033[46m測試開始！小車將使用最新大腦進行避障。按下 Ctrl+C 可停止。\033[0m")
    
    try:
        while True:
            # 使用 eval_actions 確保動作是確定的（不帶隨機探索）
            action = agent.eval_actions(obs)
            
            # 讓環境走一步
            obs, reward, terminated, truncated, _ = env.step(np.array(action))
            
            if terminated or truncated:
                print(f"回合結束，最終獎勵: {reward:.2f}")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\n停止測試。")
        env.close()

if __name__ == "__main__":
    app.run(main)