"""
獎勵計算模組

此模組提供模組化的獎勵函數，便於調整訓練信號：

獎勵組成：
1. 距離獎勵：每靠近目標 1 像素得 1.5 分
2. 時間懲罰：每步 -0.02 分（鼓勵快速完成）
3. 姿態懲罰：未正對全局目標時扣分（進房間時啟動）
4. 路徑點獎勵：到達中間路徑點 +1 分
5. 房間完成獎勵：到達房間 +20～40 分（依對準度）
6. 完全完成獎勵：8 間房間全數到達 +1000 分
7. 碰撞懲罰：碰撞障礙物 -50 分（終止回合）
"""
import math


def compute_heading_penalty(car_angle, target_pos, nose_c_x, nose_c_y, dist_to_target):
    """
    計算姿態不對齐的懲罰
    
    目的：當車子靠近房間時，強制車頭朝向房間入口
    實現：車子距房間 < 150 像素時，每差 1° 扣 0.05/180 分
    
    參數：
        car_angle: 車體朝向（度數）
        target_pos: 目標位置 (x, y)
        nose_c_x, nose_c_y: 車頭中心座標
        dist_to_target: 到目標的距離
        
    返回值：
        姿態懲罰分數
    """
    target_ang = math.degrees(math.atan2(target_pos[0]-nose_c_x, -(target_pos[1]-nose_c_y)))
    rel_ang_deg = (target_ang - car_angle + 180) % 360 - 180
    rel_ang_norm = abs(rel_ang_deg / 180.0)
    
    # Apply heading penalty only when close to room entrance
    if dist_to_target < 150.0:
        return rel_ang_norm * 0.05
    else:
        return 0.0


def compute_step_reward(prev_dist, curr_dist, heading_penalty=0.0, time_penalty=0.02):
    """
    計算每一步的基礎獎勵
    
    獎勵構成：
    reward = -time_penalty - heading_penalty + (prev_dist - curr_dist) * 1.5
    
    參數：
        prev_dist: 前一步到目標的距離
        curr_dist: 當前到目標的距離
        heading_penalty: 姿態懲罰（預設 0.0）
        time_penalty: 單步時間懲罰（預設 0.02）
        
    返回值：
        步驟獎勵
    """
    return -time_penalty - heading_penalty + (prev_dist - curr_dist) * 1.5


def get_waypoint_reward():
    """Reward for reaching intermediate waypoint."""
    return 1.0


def get_goal_room_reward(global_rel_ang_norm):
    """
    計算到達房間的獎勵（依對準度調整）
    
    實現：room_bonus = 20.0 * (1.0 - alignment_error)
    - 完全對準：+20 分
    - 歪斜 45°：+10 分
    - 歪斜 90°：0 分
    
    參數：
        global_rel_ang_norm: 歸一化的角度誤差（0.0～1.0）
        
    返回值：
        房間入場獎勵
    """
    return 20.0 * (1.0 - global_rel_ang_norm)


def get_collision_penalty():
    """Penalty for collision."""
    return -50.0


def get_rear_collision_penalty():
    """Penalty for rear bumper touching goal."""
    return -50.0


def get_episode_completion_reward():
    """Reward for completing all 8 rooms."""
    return 1000.0
