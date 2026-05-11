import os
from collections.abc import Mapping
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


def _find_first_kernel_in_dim(tree):
    """Recursively find first Dense kernel input dim from params pytree."""
    if isinstance(tree, dict):
        if "kernel" in tree and hasattr(tree["kernel"], "shape"):
            shape = tree["kernel"].shape
            if len(shape) == 2:
                return int(shape[0])
        for value in tree.values():
            found = _find_first_kernel_in_dim(value)
            if found is not None:
                return found
    return None


def _find_checkpoint_action_dim(tree):
    """Find action dim from actor params (prefer OutputDenseMean kernel shape[1])."""
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            if key == "OutputDenseMean" and isinstance(value, Mapping):
                kernel = value.get("kernel", None)
                if hasattr(kernel, "shape") and len(kernel.shape) == 2:
                    return int(kernel.shape[1])

            found = _find_checkpoint_action_dim(value)
            if found is not None:
                return found
    return None


def _checkpoint_step(path: str) -> int:
    """Extract step number from path like .../checkpoint_350000."""
    base = os.path.basename(path.rstrip("/"))
    if base.startswith("checkpoint_"):
        try:
            return int(base.split("_")[-1])
        except ValueError:
            return -1
    return -1


def _find_default_checkpoint_dir() -> str:
    """Find best checkpoint directory when --checkpoint_dir is not provided."""
    candidates = [
        "checkpoints",
        os.path.join("rlpd", "s42_online", "checkpoints"),
        os.path.join("rlpd", "s42_online(六個移動)", "checkpoints"),
        os.path.join("rlpd", "s42_online(固定旋轉)", "checkpoints"),
        os.path.join("rlpd", "s42_0pretrain", "checkpoints"),
    ]

    best_dir = ""
    best_step = -1

    for ckpt_dir in candidates:
        abs_dir = os.path.abspath(ckpt_dir)
        if not os.path.isdir(abs_dir):
            continue

        try:
            entries = [
                os.path.join(abs_dir, name)
                for name in os.listdir(abs_dir)
                if name.startswith("checkpoint_") and os.path.isdir(os.path.join(abs_dir, name))
            ]
        except OSError:
            continue

        if not entries:
            continue

        step = max(_checkpoint_step(p) for p in entries)
        if step > best_step:
            best_step = step
            best_dir = abs_dir

    return best_dir

def main(_):
    # 若沒指定，嘗試自動尋找可用的 checkpoint 目錄
    if not FLAGS.checkpoint_dir:
        FLAGS.checkpoint_dir = _find_default_checkpoint_dir()
        if FLAGS.checkpoint_dir:
            print(f"\033[43m未指定 --checkpoint_dir，自動使用: {FLAGS.checkpoint_dir}\033[0m")

    if not FLAGS.checkpoint_dir:
        print("\033[41m找不到可用 checkpoint。請指定 --checkpoint_dir（例如: rlpd/s42_online/checkpoints）\033[0m")
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

    # 4.05 自動對齊 action 維度（處理舊 checkpoint 與新環境動作數不一致）
    checkpoint_action_dim = _find_checkpoint_action_dim(agent.actor.params)
    if checkpoint_action_dim is None:
        checkpoint_action_dim = action_dim

    if checkpoint_action_dim != action_dim:
        print(
            f"\033[43m警告：checkpoint 期望 action 維度={checkpoint_action_dim}，"
            f"目前環境 action 維度={action_dim}，將自動映射。\033[0m"
        )
        agent = globals()[model_cls].create(FLAGS.seed, observation_dim, checkpoint_action_dim, **kwargs)
        agent = checkpoints.restore_checkpoint(latest_ckpt, agent)

    # 4.1 自動對齊 observation 維度（處理舊 checkpoint 與新環境特徵數不一致）
    expected_obs_dim = _find_first_kernel_in_dim(agent.actor.params)
    if expected_obs_dim is None:
        expected_obs_dim = observation_dim
    if expected_obs_dim != observation_dim:
        print(
            f"\033[43m警告：checkpoint 期望 obs 維度={expected_obs_dim}，"
            f"目前環境 obs 維度={observation_dim}，將自動對齊後再推論。\033[0m"
        )

    # 5. 開始表演
    obs, _ = env.reset()
    print("\033[46m測試開始！小車將使用最新大腦進行避障。按下 Ctrl+C 可停止。\033[0m")
    
    try:
        while True:
            # 使用 eval_actions 確保動作是確定的（不帶隨機探索）
            policy_obs = obs
            if expected_obs_dim < observation_dim:
                policy_obs = obs[:expected_obs_dim]
            elif expected_obs_dim > observation_dim:
                pad = np.zeros((expected_obs_dim - observation_dim,), dtype=np.float32)
                policy_obs = np.concatenate([obs, pad], axis=0)

            action = np.array(agent.eval_actions(policy_obs), dtype=np.float32)

            # 將策略動作映射到環境動作維度
            if checkpoint_action_dim < action_dim:
                env_action = np.zeros((action_dim,), dtype=np.float32)
                env_action[:checkpoint_action_dim] = action[:checkpoint_action_dim]
            elif checkpoint_action_dim > action_dim:
                env_action = action[:action_dim]
            else:
                env_action = action
            
            # 讓環境走一步
            obs, reward, terminated, truncated, _ = env.step(env_action)
            
            if terminated or truncated:
                print(f"回合結束，最終獎勵: {reward:.2f}")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\n停止測試。")
        env.close()

if __name__ == "__main__":
    app.run(main)