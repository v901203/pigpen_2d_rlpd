"""
感測器與觀測模組

此模組提供：
1. LIDAR 感測器：模擬 360° 環形雷射測距（36 道射線）
2. 觀測計算：將感測資訊轉換為神經網路輸入向量

觀測向量構成（40D）：
- 36D：LIDAR 距離測量
- 1D：到局部目標（藍點）的相對距離
- 1D：到局部目標的相對角度
- 1D：到全局目標（紅點）的相對距離
- 1D：到全局目標的相對角度
"""
import numpy as np
import math


class LidarSensor:
    """LIDAR sensor simulation."""
    
    def __init__(self, num_rays=36, lidar_range=250.0):
        """
        初始化 LIDAR 感測器
        
        參數：
            num_rays: 雷射射線數（預設 36 條，每 10 度一條）
            lidar_range: 最大測量距離（像素）
        """
        self.num_rays = num_rays
        self.lidar_range = lidar_range
        self.hits = []

    def scan(self, car_x, car_y, car_angle, dynamic_mask, screen_w, screen_h):
        """
        執行 LIDAR 掃描
        
        原理：從車體中心發射 36 道雷射，每 5 像素檢查一次是否碰撞障礙物
        輸出：歸一化距離（0.0-1.0）與碰撞點座標
        
        參數：
            car_x, car_y: 車體中心座標
            car_angle: 車體朝向（度數）
            dynamic_mask: pygame 碰撞遮罩
            screen_w, screen_h: 螢幕尺寸
            
        返回值：
            (lidar_readings, hit_positions) 元組
            - lidar_readings：36 個歸一化距離值
            - hit_positions：36 個碰撞點座標
        """
        lidar = []
        self.hits = []
        
        for i in range(self.num_rays):
            rad = math.radians(car_angle + i*(360/self.num_rays))
            dx, dy = math.sin(rad), -math.cos(rad)
            d_res = self.lidar_range
            hit_pos = (car_x + dx * self.lidar_range, car_y + dy * self.lidar_range)
            
            for d in range(0, int(self.lidar_range), 5):
                px, py = int(car_x + dx*d), int(car_y + dy*d)
                if not (0<=px<screen_w and 0<=py<screen_h) or dynamic_mask.get_at((px,py)):
                    d_res = d
                    hit_pos = (px, py)
                    break
                    
            lidar.append(d_res / self.lidar_range)
            self.hits.append(hit_pos)
        
        return lidar, self.hits


def compute_target_observation(car_x, car_y, nose_c_x, nose_c_y, car_angle, 
                               target_pos, is_tracking_final, screen_h):
    """
    計算局部目標觀測（紅點或藍點）
    
    參考點推轉：
    - 不是最終目標時（is_tracking_final=False）：以車體中心作为參考點
    - 是最終目標時（is_tracking_final=True）：以車頭中心作为參考點（精確較正）
    
    參數：
        car_x, car_y: 車體中心座標
        nose_c_x, nose_c_y: 車頭中心座標
        car_angle: 車體朝向（度數）
        target_pos: 目標位置 (x, y)
        is_tracking_final: 是否追蹤最終目標
        screen_h: 螢幕高度（施一化分訊）
        
    返回值：
        (相對距離, 相對角度) 元組，均歸一化至 [-1, 1]
    """
    ref_x = nose_c_x if is_tracking_final else car_x
    ref_y = nose_c_y if is_tracking_final else car_y

    rel_dist = min(math.hypot(target_pos[0]-ref_x, target_pos[1]-ref_y)/screen_h, 1.0)
    target_ang = math.degrees(math.atan2(target_pos[0]-ref_x, -(target_pos[1]-ref_y)))
    rel_ang = ((target_ang - car_angle + 180) % 360 - 180) / 180.0
    
    return rel_dist, rel_ang


def compute_global_goal_observation(nose_c_x, nose_c_y, car_angle, 
                                    global_goal_pos, screen_h):
    """
    計算全局目標觀測（紅點 GPS）
    
    目的：讓 AI 一一直看得到最終橫点（紅點）
    - 燕上對準 ⇒ 扣分（姿態懲罰）
    - 路徑觀測上崩漫（藍點）時不會沙滲
    
    參數：
        nose_c_x, nose_c_y: 車頭中心座標
        car_angle: 車體朝向（度數）
        global_goal_pos: 全局目標位置（紅點）
        screen_h: 螢幕高度（施一化分訊）
        
    返回值：
        (全局距離, 全局角度) 元組，均歸一化至 [-1, 1]
    """
    global_dist = min(math.hypot(global_goal_pos[0]-nose_c_x, global_goal_pos[1]-nose_c_y)/screen_h, 1.0)
    global_ang = math.degrees(math.atan2(global_goal_pos[0]-nose_c_x, -(global_goal_pos[1]-nose_c_y)))
    global_rel_ang = ((global_ang - car_angle + 180) % 360 - 180) / 180.0
    
    return global_dist, global_rel_ang


def build_observation(lidar_readings, local_dist, local_ang, global_dist, global_ang):
    """
    流汷永上幼觀測向量（40D）
    
    向量構成：
    [36D LIDAR] + [1D紅點距離] + [1D紅點角度] + [1D藍點距離] + [1D藍點角度]
    = 40D
    
    參數：
        lidar_readings: 36 個 LIDAR 測量值
        local_dist, local_ang: 局部目標（藍點）觀測
        global_dist, global_ang: 全局目標（紅點）觀測
        
    返回值：
        40D numpy 陣列
    """
    return np.array(lidar_readings + [local_dist, local_ang, global_dist, global_ang], 
                    dtype=np.float32)
