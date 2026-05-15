#!/home/vito/Desktop/pig_pen_2d_rl/rl_env/bin/python

import numpy as np
import pygame
from pig_pen_env import PigPenEnv

def main():
    # 這裡門寬不要放太大，用真實比例 (1.0) 錄製最好的示範
    env = PigPenEnv(render_mode="human", door_width_scale=1.0)
    obs, _ = env.reset()
    
    print("🎥 開始錄製專家示範！")
    print("請用方向鍵控制小車跑完 8 間豬舍。按 ESC 可以提早結束錄製並存檔。")
    print("按空白鍵開始錄製（SPACE）。")
    
    expert_obs = []
    expert_actions = []
    expert_rewards = []
    expert_next_obs = []
    expert_dones = []
    
    # 等待使用者按下空白鍵開始錄製（避免視窗未聚焦造成按鍵遺失）
    running = True
    started = False
    while not started:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    started = True
                    break
        # 在等待期間保持環境更新，讓畫面可見
        obs, _, terminated, truncated, _ = env.step([0.0, 0.0])
        if terminated or truncated:
            obs, _ = env.reset()

    # 開始正式錄製
    # 使用 KEYDOWN/KEYUP 狀態追蹤，較可靠於視窗聚焦與重複按鍵情況
    pressed = {"up": False, "down": False, "left": False, "right": False}
    while running:
        action = [0.0, 0.0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # 可按 SPACE 暫停錄製（選擇性）
                    pass
                elif event.key in (pygame.K_UP, pygame.K_w):
                    pressed["up"] = True
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    pressed["down"] = True
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    pressed["left"] = True
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    pressed["right"] = True
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_UP, pygame.K_w):
                    pressed["up"] = False
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    pressed["down"] = False
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    pressed["left"] = False
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    pressed["right"] = False

        # 根據按鍵狀態組成動作
        if pressed["up"]:
            action[0] = 1.0
        elif pressed["down"]:
            action[0] = -1.0
        if pressed["left"]:
            action[1] = -1.0
        elif pressed["right"]:
            action[1] = 1.0
        
        # 🌟 核心過濾邏輯：只有當你有按按鍵時，才把資料錄下來
        # 避免錄到一堆「停在原地發呆」的資料，導致 AI 學會偷懶
        if action[0] != 0.0 or action[1] != 0.0:
            expert_obs.append(obs)
            expert_actions.append(action)
            
            # 執行動作並收集完整的步驟資料
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            expert_rewards.append(reward)
            expert_next_obs.append(next_obs)
            expert_dones.append(done)
            
            obs = next_obs
            
            if done:
                print(f"✅ 回合結束！已收集 {len(expert_obs)} 筆專家資料。")
                break
        else:
            # 空動作也要執行環境步驟，但不記錄
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"✅ 回合結束！已收集 {len(expert_obs)} 筆專家資料。")
                break
            
    env.close()
    
    # 將完整資料存成 Numpy 壓縮檔
    if len(expert_obs) > 0:
        np.savez(
            "expert_data.npz", 
            obs=np.array(expert_obs), 
            actions=np.array(expert_actions),
            rewards=np.array(expert_rewards),
            next_obs=np.array(expert_next_obs),
            dones=np.array(expert_dones)
        )
        print(f"💾 錄製完成！已將 {len(expert_obs)} 步的完整資料存至 expert_data.npz")
    else:
        print("⚠️ 沒有錄製到任何動作。")

if __name__ == "__main__":
    main()