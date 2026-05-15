#!/home/vito/Desktop/pig_pen_2d_rl/rl_env/bin/python
"""
豬舍 2D 導航環境 - 模組化版本

強化學習環境定義：一個自主小車在 2D 豬舍中導航
- 靜止障礙：8 間豬舍房間（含門與飼料桶）
- 動態障礙：16 隻隨機遊走的豬隻
- 任務：順序訪問所有 8 間豬舍並停靠餵食
- 觀測：36 道 LIDAR 射線 + 局部目標 + 全局 GPS
- 動作：前進速度 + 轉向角速度（兩維連續）
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

# Import modularized components
from rlpd.envs.pig import Pig
from rlpd.envs.navigation import NavigationGrid, generate_global_goals, draw_static_obstacles
from rlpd.envs.sensors import LidarSensor, compute_target_observation, compute_global_goal_observation, build_observation
from rlpd.envs.physics import CarPhysics, render_car
from rlpd.envs.rewards import (
    compute_heading_penalty, compute_step_reward, get_waypoint_reward,
    get_goal_room_reward, get_collision_penalty, get_rear_collision_penalty,
    get_episode_completion_reward
)


class PigPenEnv(gym.Env):
    """
    豬舍 2D 導航環境
    
    繼承 gymnasium.Env，實現完整 RL 環境介面。
    包含靜態環境構建、動態物體更新、獎勵計算、渲染等功能。
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, door_width_scale=1.0):
        """
        初始化豬舍環境
        
        參數：
            render_mode: "human" 用於可視化，None 用於無頭模式
            door_width_scale: 門寬度縮放係數（影響難度，1.0 = 標準寬度）
        
        初始化步驟：
        1. 設置場景尺寸（600x800 像素）
        2. 初始化 pygame 並創建顯示窗口（如果需要）
        3. 建立靜態環境（牆壁、門、飼料桶）
        4. 構建碰撞網格用於路徑規劃
        5. 生成 8 個豬舍目標位置
        6. 初始化 16 隻豬隻
        """
        super().__init__()
        self.render_mode = render_mode
        
        # 🌟 環境尺寸參數
        self.scale = 0.6  # 縮放係數（原始尺寸 × 0.6）
        self.screen_w, self.screen_h = 600, 800  # 畫面解析度（像素）
        self.center_x = self.screen_w // 2  # 中心走道 x 座標
        
        # 🚗 小車尺寸（根據縮放係數計算）
        self.car_w, self.car_h = 40 * self.scale, 45 * self.scale
        
        # 🏠 豬舍房間配置
        self.pen_w = 275 * self.scale  # 房間寬度
        self.pen_h = 289 * self.scale  # 房間高度
        self.corridor_w = 150 * self.scale  # 中央走道寬度
        self.door_w = 59 * self.scale * door_width_scale  # 房間出入口寬度（可調難度）
        self.bin_r = (60 / 2) * self.scale  # 飼料桶半徑（餵食目標） 
        
        # 📡 感測器配置
        self.num_rays = 36  # LIDAR 射線數（每 10° 一道）
        self.lidar_range = 250.0  # 最大感測距離（像素）
        
        # 🎮 動作與觀測空間
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # action[0]: 線性速度 [-1, 1] → [-5, 5] 像素/幀
        # action[1]: 角速度 [-1, 1] → [-5, 5] 度/幀
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(40,), dtype=np.float32)
        # obs 組成（40D）：
        # [0:36]   = LIDAR 距離 (36 道)
        # [36]     = 局部目標距離（藍點）
        # [37]     = 局部目標角度（藍點）
        # [38]     = 全局目標距離（紅點）
        # [39]     = 全局目標角度（紅點）
        
        # 🎭 初始化 pygame 渲染系統
        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("Pig Pen Navigation")
            self.clock = pygame.time.Clock()  # 用於控制往率（30 fps）
        else:
            pygame.init()
            self.window = pygame.Surface((self.screen_w, self.screen_h))  # 虫擬表面（無顯示）
        
        # 🗣️ 初始化環境層級
        self.obstacle_surface = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        # 靜態障礙層（牆壁、門、飼料桶） - 不會改變
        
        self.dynamic_surface = None  # 動態層（豬隻） - 每幀更新
        self.dynamic_mask = None  # 碰撞遮罩 - 用於高效碰撞檢測
        
        # 🧄 初始化模組化組件
        self.car_physics = CarPhysics(self.car_w, self.car_h)  # 小車運動学
        self.lidar_sensor = LidarSensor(self.num_rays, self.lidar_range)  # LIDAR 掃描器
        self.nav_grid = NavigationGrid(self.screen_w, self.screen_h, cell_size=5)  # 路徑規劃網格
        
        # 🎨 構建靜態環境（一次性，不會改變）
        self.pen_info = draw_static_obstacles(
            self.obstacle_surface, self.screen_w, self.screen_h,
            self.center_x, self.corridor_w, self.pen_w, self.pen_h,
            self.door_w, self.bin_r
        )
        # pen_info 包含 8 間房間的邊界與飼料桶位置
        
        # 🗯️ 構建碰撞檢測網格（用於 A* 路徑規劃）
        self.nav_grid.build_from_obstacle_surface(self.obstacle_surface, self.car_w, self.car_h)
        
        # 🎯 生成目標與動態物體
        self.global_goals = generate_global_goals(self.center_x, self.corridor_w, self.pen_w, self.pen_h)
        # 8 個目標按順序排列（房間 0-7 的飼料桶位置）
        
        self.pigs = []  # 動態豬隻列表（reset() 時初始化）
        
        # 📈 狀態追蹤變數
        self.steps = 0  # 當前回合步數
        self.current_goal_idx = 0  # 當前目標房間索引（0-7）
        self.local_waypoints = []  # 剩餘藍色路徑點
        self.target_pos = None  # 當前目標座標（藍點或紅點）
        self.prev_dist = 0.0  # 前一步到目標的距離（用於獎勵計算）

    def _update_dynamic(self):
        """更新動態障礙層（豬隻）與碰撞遮罩
        
        步驟：
        1. 複製靜態障礙層
        2. 在上面繪製所有豬隻
        3. 轉換為 pygame 遮罩用於高效碰撞檢測
        
        此方法需在每一幀前呼叫以保持碰撞檢測的準確性
        """
        self.dynamic_surface = self.obstacle_surface.copy()
        for p in self.pigs:
            p.draw(self.dynamic_surface)
        self.dynamic_mask = pygame.mask.from_surface(self.dynamic_surface)

    def reset(self, seed=None, options=None):
        """重置環境到初始狀態
        
        步驟：
        1. 重置小車位置（中心走道下方）、朝向（0°）、步數
        2. 重置目標追蹤（目標 0、無路徑點）
        3. 計算到初始目標的距離
        4. 生成 16 隻豬隻（2 隻/房間 × 8 間房間）
        5. 更新碰撞層
        
        返回值：
            (obs, info) - 初始觀測向量與空字典
        """
        super().reset(seed=seed)
        
        # 🌟 重置小車狀態
        # 🎲 x 軸隨機化：出生點左右變化（增加初始多樣性）
        spawn_x_min, spawn_x_max = 100, 500  # x 軸範圍（像素）
        self.car_physics.car_x = self.np_random.uniform(spawn_x_min, spawn_x_max)  # 隨機 x 位置
        self.car_physics.car_y = 750.0  # 垂直位置固定：下方起始區域
        self.car_physics.car_angle = 0.0  # 初始朝向：向上（0°）
        self.car_physics.update_car_points()  # 更新碰撞檢測點
        
        self.steps = 0  # 重置步數計數器
        self.current_goal_idx = 0  # 從目標 0 開始
        
        # Initialize path planning
        self.local_waypoints = self.nav_grid.a_star(
            (self.car_physics.car_x, self.car_physics.car_y),
            self.global_goals[self.current_goal_idx]
        )
        
        if len(self.local_waypoints) > 0:
            self.target_pos = self.local_waypoints.pop(0)
        else:
            self.target_pos = self.global_goals[self.current_goal_idx]
        
        # Initialize distance tracking
        is_tracking_final = len(self.local_waypoints) == 0
        ref_x = self.car_physics.nose_c_x if is_tracking_final else self.car_physics.car_x
        ref_y = self.car_physics.nose_c_y if is_tracking_final else self.car_physics.car_y
        self.prev_dist = math.hypot(ref_x - self.target_pos[0], ref_y - self.target_pos[1])
        
        # Initialize pigs
        self.pigs = []
        pid = 0
        for info in self.pen_info:
            bx, by = info["bin_pos"]
            safe_spawn_points = [(bx - 35, by - 40), (bx + 35, by + 40)]
            for pos in safe_spawn_points:
                self.pigs.append(Pig(pos, info["bounds"], info["bin_pos"], 
                                    self.bin_r, self.scale, pid))
                pid += 1
        
        self._update_dynamic()
        return self._get_obs(), {}

    def step(self, action):
        """執行一個環境時間步
        
        整體流程：
        1. 📊 更新狀態
           - 豬隻行為與位置
           - 動態障礙層
           - 小車位置與方向
        
        2. 🎯 碰撞檢測與距離計算
           - 與靜態/動態障礙的碰撞
           - 保險桿到目標的距離
           - 姿態與目標的偏差
        
        3. 💰 獎勵計算
           - 基礎獎勵：距離進度 × 1.5
           - 時間懲罰：-0.02/步
           - 姿態懲罰：靠近房間時強制對齐
           - 碰撞懲罰：-50（終止）
           - 路徑點獎勵：+1
           - 房間完成獎勵：+20~40
           - 全部完成獎勵：+1000（終止）
        
        4. 🎨 渲染（如果需要）
        
        返回值：
            (obs, reward, terminated, truncated, info)
        """
        self.steps += 1  # 增加步數
        
        # 📊 更新動態狀態（豬隻 + 小車 + 碰撞層）
        for p in self.pigs:
            p.update(self.pigs)  # 更新每隻豬的行為與位置
        self._update_dynamic()  # 更新動態障礙層與碰撞遮罩
        
        # 🚗 更新小車玩家動作（前進/旋轉）
        self.car_physics.step(action)
        
        # 🎨 碰撞檢測
        is_collision = self.car_physics.check_collision(self.dynamic_mask)
        
        # 📏 計算距離（用於獎勵不同步驟）
        is_tracking_final = len(self.local_waypoints) == 0  # 是否追蹤最終目標（紅點）
        trigger_dist = 15.0 if is_tracking_final else 30.0  # 觸發觀實戰距離
        
        ref_x = self.car_physics.nose_c_x if is_tracking_final else self.car_physics.car_x  # 參考點（根據是否是最終目標）
        ref_y = self.car_physics.nose_c_y if is_tracking_final else self.car_physics.car_y
        dist_ref = math.hypot(ref_x - self.target_pos[0], ref_y - self.target_pos[1])  # 到當前目標的距離
        
        min_front_dist, min_rear_dist = self.car_physics.compute_bumper_distances(self.target_pos)  # 保險桿距離
        dist_c = math.hypot(self.car_physics.car_x - self.target_pos[0], 
                           self.car_physics.car_y - self.target_pos[1])
        min_any_dist = min(dist_c, min_front_dist, min_rear_dist)  # 最小距離（任何接觸點）
        
        # 🎯 計算全局目標（邊界檢查：如果已完成所有目標，使用最後一個）
        goal_idx = min(self.current_goal_idx, len(self.global_goals) - 1)
        current_global_goal = self.global_goals[goal_idx]
        dist_to_global = math.hypot(self.car_physics.car_x - current_global_goal[0], 
                                   self.car_physics.car_y - current_global_goal[1])
        
        global_ang = math.degrees(math.atan2(
            current_global_goal[0] - self.car_physics.nose_c_x, 
            -(current_global_goal[1] - self.car_physics.nose_c_y)
        ))
        global_rel_ang_deg = (global_ang - self.car_physics.car_angle + 180) % 360 - 180  # 相對角度（正規化到 [-180, 180]）
        global_rel_ang_norm = abs(global_rel_ang_deg / 180.0)  # 正規化相對角度 [0, 1]
        
        # 🎯 姿態對齊懲罰（靠近房間時強制正確朝向）
        heading_penalty = 0.05 * global_rel_ang_norm if dist_to_global < 150.0 else 0.0
        
        # 💰 計算獎勵
        reward = compute_step_reward(self.prev_dist, dist_ref, heading_penalty)  # 基礎獎勵 = 距離進度 - 時間懲罰 - 姿態懲罰
        self.prev_dist = dist_ref  # 更新參考距離用於下一步
        
        terminated = False  # 環境終止標記
        
        # 💥 碰撞檢測
        if is_collision:
            reward += get_collision_penalty()  # 碰撞懲罰 -50
            terminated = True  # 終止回合
        
        # 🏠 目標互動處理
        elif is_tracking_final:  # 已追蹤到最終目標（紅點）
            if min_rear_dist < trigger_dist:
                reward += get_rear_collision_penalty()  # 後保險桿接觸懲罰
                terminated = True  # 失敗
                print("🚨 警告：後保險桿接觸目標！失敗！")
            elif min_front_dist < trigger_dist:  # 前保險桿接觸目標
                reward += get_goal_room_reward(global_rel_ang_norm)  # 房間完成獎勵 20-40
                self.current_goal_idx += 1  # 進到下一個房間
                
                if self.current_goal_idx >= len(self.global_goals):
                    reward += get_episode_completion_reward()
                    terminated = True
                else:
                    self.local_waypoints = self.nav_grid.a_star(
                        (self.car_physics.car_x, self.car_physics.car_y),
                        self.global_goals[self.current_goal_idx]
                    )
                    if len(self.local_waypoints) > 0:
                        self.target_pos = self.local_waypoints.pop(0)
                    else:
                        self.target_pos = self.global_goals[self.current_goal_idx]
                    
                    is_tracking_final_new = len(self.local_waypoints) == 0
                    new_ref_x = self.car_physics.nose_c_x if is_tracking_final_new else self.car_physics.car_x
                    new_ref_y = self.car_physics.nose_c_y if is_tracking_final_new else self.car_physics.car_y
                    self.prev_dist = math.hypot(new_ref_x - self.target_pos[0], 
                                               new_ref_y - self.target_pos[1])
        else:  # 還在追蹤藍色路徑點
            if min_any_dist < trigger_dist:  # 接觸藍點
                reward += get_waypoint_reward()  # 路徑點獎勵 +1
                self.target_pos = self.local_waypoints.pop(0)  # 取下一個路徑點
                
                # 重新計算參考距離
                is_tracking_final_new = len(self.local_waypoints) == 0
                new_ref_x = self.car_physics.nose_c_x if is_tracking_final_new else self.car_physics.car_x
                new_ref_y = self.car_physics.nose_c_y if is_tracking_final_new else self.car_physics.car_y
                self.prev_dist = math.hypot(new_ref_x - self.target_pos[0], 
                                           new_ref_y - self.target_pos[1])
        
        # 🎨 渲染（如果懲罰模式是 human）
        if self.render_mode == "human":
            self.render()
        
        # ⚠️ 終止條件
        truncated = self.steps >= 5000  # 最多 5000 步
        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self):
        """構建完整的觀測向量（40D）
        
        觀測組成：
        1. LIDAR 掃描（36D）
           - 36 道射線均勻分佈 360°
           - 每道射線單位化距離 [0.0, 1.0]
        
        2. 局部目標觀測（2D）
           - 相對距離：到藍色路徑點的正規化距離
           - 相對角度：車頭與藍點方向的夾角 [-180°, 180°]
           - 用於局部導航
        
        3. 全局目標觀測（2D）
           - 相對距離：到紅色飼料桶的正規化距離
           - 相對角度：車頭與紅點方向的夾角 [-180°, 180°]
           - 用於長期導向與姿態對齊
        
        全部轉換為 [-1.0, 1.0] 範圍的 40D 向量
        """
        # 📡 LIDAR 掃描（36 道射線）
        lidar_readings, _ = self.lidar_sensor.scan(
            self.car_physics.car_x, self.car_physics.car_y, self.car_physics.car_angle,
            self.dynamic_mask, self.screen_w, self.screen_h
        )
        
        # 🎯 局部目標觀測（藍色路徑點）
        is_tracking_final = len(self.local_waypoints) == 0  # 檢查是否到達最後一個路徑點
        local_dist, local_ang = compute_target_observation(
            self.car_physics.car_x, self.car_physics.car_y,
            self.car_physics.nose_c_x, self.car_physics.nose_c_y,
            self.car_physics.car_angle, self.target_pos, is_tracking_final, self.screen_h
        )
        
        # 🗺️ 全局目標觀測（紅色飼料桶，邊界檢查）
        # 如果已完成所有目標，使用最後一個目標位置
        goal_idx = min(self.current_goal_idx, len(self.global_goals) - 1)
        current_global_goal = self.global_goals[goal_idx]
        global_dist, global_ang = compute_global_goal_observation(
            self.car_physics.nose_c_x, self.car_physics.nose_c_y,
            self.car_physics.car_angle, current_global_goal, self.screen_h
        )
        
        # 🎨 組合 40D 觀測向量：[36D LIDAR + 2D 局部 + 2D 全局]
        return build_observation(lidar_readings, local_dist, local_ang, global_dist, global_ang)

    def render(self):
        """渲染環境可視化
        
        渲染層次（從下到上）：
        1. 白色背景
        2. 靜態障礙層（黑色牆壁、綠色飼料桶）
        3. 動態層（豬隻）
        4. 路徑規劃網格（紅色半透明 - 調試用）
        5. 路徑可視化（藍線 + 藍點）
        6. 目標點（紅點）
        7. LIDAR 射線與打點（淺藍線 + 橙點）
        8. 小車（藍色正方形 + 青色指向燈）
        9. 保險桿（黃色前3點 + 綠色後3點）
        
        更新幀率：30 fps（由 self.clock 控制）
        """
        if self.render_mode != "human":
            return  # 僅在 render_mode="human" 時執行
        
        # 🎨 清潔背景
        self.window.fill((255, 255, 255))  # 白色背景
        
        # 🏗️ 繪製靜態與動態層
        self.window.blit(self.dynamic_surface, (0, 0))  # 靜態 + 豬隻層
        self.window.blit(self.nav_grid.debug_grid_surface, (0, 0))  # 路徑規劃網格（紅色）
        
        # 🚗 渲染小車、路徑、感測器
        render_car(self.window, self.car_physics, self.lidar_sensor.hits,
                  self.target_pos, self.local_waypoints)
        
        # 🖥️ 更新顯示
        pygame.display.flip()  # 將繪製內容顯示到窗口
        self.clock.tick(30)  # 30 fps 更新頻率

    def close(self):
        """清理資源
        
        關閉 pygame 窗口並釋放相關資源
        """
        if self.render_mode == "human":
            pygame.quit()


if __name__ == "__main__":
    """🎮 互動模式 - 使用鍵盤操控小車
    
    用途：測試環境、調試、觀察行為
    
    控制方式：
    - ↑ / W：前進
    - ↓ / S：後退
    - ← / A：左轉
    - → / D：右轉
    - ESC 或關閉窗口：退出
    """
    env = PigPenEnv(render_mode="human")
    obs, _ = env.reset()
    print("🎮 互動模式 - 用方向鍵 WASD 控制小車，按 ESC 或關閉窗口退出")
    
    while True:
        act = [0.0, 0.0]
        for e in pygame.event.get(): 
            if e.type == pygame.QUIT: 
                pygame.quit()
                exit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: act[0] = 1.0
        if keys[pygame.K_DOWN]: act[0] = -1.0
        if keys[pygame.K_LEFT]: act[1] = -1.0
        if keys[pygame.K_RIGHT]: act[1] = 1.0
        obs, reward, done, trunc, _ = env.step(act)
        if done or trunc:
            print("✅ 回合結束，重新開始")
            obs, _ = env.reset()
