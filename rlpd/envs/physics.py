"""
車體物理與渲染模組

此模組提供：
1. 車體運動學：速度/角速度 → 位置/方向更新
2. 碰撞檢測：檢測車體與動態障礙物的碰撞
3. 保險桿感測器：前後各 3 個感測點（用於精確碰撞檢測）
4. 渲染：繪製車體、感測器點、LIDAR 射線、目標路徑
"""
import pygame
import math


class CarPhysics:
    """Car kinematics and collision detection."""
    
    def __init__(self, car_w, car_h):
        """
        初始化車體物理模型
        
        參數：
            car_w: 車體寬度
            car_h: 車體長度
        """
        self.car_w = car_w
        self.car_h = car_h
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_angle = 0.0
        
        # Bumper sensor positions
        self.nose_c_x = 0.0
        self.nose_c_y = 0.0
        self.nose_r_x = 0.0
        self.nose_r_y = 0.0
        self.nose_l_x = 0.0
        self.nose_l_y = 0.0
        
        self.rear_c_x = 0.0
        self.rear_c_y = 0.0
        self.rear_r_x = 0.0
        self.rear_r_y = 0.0
        self.rear_l_x = 0.0
        self.rear_l_y = 0.0

    def update_car_points(self):
        """根據車體姿態更新保險桿感測點位置
        
        前保險桿（黃色）：車頭中心、左、右各一點
        後保險桿（綠色）：車尾中心、左、右各一點
        用於精確計算車體與目標點的距離
        """
        rad = math.radians(self.car_angle)
        fw_x = math.sin(rad)
        fw_y = -math.cos(rad)
        rt_x = math.cos(rad)
        rt_y = math.sin(rad)

        # Front bumpers
        self.nose_c_x = self.car_x + (self.car_h / 2) * fw_x
        self.nose_c_y = self.car_y + (self.car_h / 2) * fw_y
        self.nose_r_x = self.nose_c_x + (self.car_w / 2 - 2) * rt_x
        self.nose_r_y = self.nose_c_y + (self.car_w / 2 - 2) * rt_y
        self.nose_l_x = self.nose_c_x - (self.car_w / 2 - 2) * rt_x
        self.nose_l_y = self.nose_c_y - (self.car_w / 2 - 2) * rt_y

        # Rear bumpers
        self.rear_c_x = self.car_x - (self.car_h / 2) * fw_x
        self.rear_c_y = self.car_y - (self.car_h / 2) * fw_y
        self.rear_r_x = self.rear_c_x + (self.car_w / 2 - 2) * rt_x
        self.rear_r_y = self.rear_c_y + (self.car_w / 2 - 2) * rt_y
        self.rear_l_x = self.rear_c_x - (self.car_w / 2 - 2) * rt_x
        self.rear_l_y = self.rear_c_y - (self.car_w / 2 - 2) * rt_y

    def step(self, action):
        """
        更新車體位置與方向
        
        運動模型：前向速度與角速度
        - action[0]：線性速度（-1~1 對應 -5~5 像素/幀）
        - action[1]：角速度（-1~1 對應 -5~5 度/幀）
        
        參數：
            action: [velocity, angular_velocity] 歸一化至 [-1, 1]
        """
        v = max(-1, min(1, action[0])) * 5.0
        omega = max(-1, min(1, action[1])) * 5.0
        
        self.car_angle += omega
        rad = math.radians(self.car_angle)
        self.car_x += v * math.sin(rad)
        self.car_y -= v * math.cos(rad)
        
        self.update_car_points()

    def check_collision(self, dynamic_mask):
        """
        檢測與動態障礙物的碰撞
        
        方法：使用 pygame mask 位元遮罩進行高效碰撞檢測
        包括：豬隻、牆壁、飼料桶等所有動態元素
        
        參數：
            dynamic_mask: pygame 碰撞遮罩
            
        返回值：
            若檢測到碰撞返回 True，否則 False
        """
        car_s = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        car_s.fill((0,0,255,255))
        rot_car = pygame.transform.rotate(car_s, -self.car_angle)
        rect = rot_car.get_rect(center=(int(self.car_x), int(self.car_y)))
        return dynamic_mask.overlap(pygame.mask.from_surface(rot_car), (rect.x, rect.y)) is not None

    def compute_bumper_distances(self, target_pos):
        """
        計算保險桿感測點到目標的距離
        
        目的：精確判定車體是否到達目標點
        策略：取前保險桿3點的最小距離與後保險桿3點的最小距離
        
        參數：
            target_pos: (x, y) 目標位置
            
        返回值：
            (front_min_dist, rear_min_dist) 元組
        """
        dist_n_c = math.hypot(self.nose_c_x - target_pos[0], self.nose_c_y - target_pos[1])
        dist_n_l = math.hypot(self.nose_l_x - target_pos[0], self.nose_l_y - target_pos[1])
        dist_n_r = math.hypot(self.nose_r_x - target_pos[0], self.nose_r_y - target_pos[1])
        min_front_dist = min(dist_n_c, dist_n_l, dist_n_r)

        dist_r_c = math.hypot(self.rear_c_x - target_pos[0], self.rear_c_y - target_pos[1])
        dist_r_l = math.hypot(self.rear_l_x - target_pos[0], self.rear_l_y - target_pos[1])
        dist_r_r = math.hypot(self.rear_r_x - target_pos[0], self.rear_r_y - target_pos[1])
        min_rear_dist = min(dist_r_c, dist_r_l, dist_r_r)
        
        return min_front_dist, min_rear_dist


def render_car(window, car_physics, lidar_hits, target_pos, local_waypoints):
    """
    渲染車體、感測器與路徑
    
    渲染統計：
    - 路徑：藍涪線檢查上一個路徑點
    - 局部路徑點：藍色圓圈（R:0, G:150, B:255）
    - 紅色點：当前目標（紅點）
    - LIDAR 射線：浅藍色線条 + 樹榥點
    - 車體：藍色正方形 + 青綠色挪臥（及時朝向）
    - 保險桿：黃色前3點 + 綠色後3點
    
    參數：
        window: pygame 蛤幕物件
        car_physics: CarPhysics 實例
        lidar_hits: LIDAR 螋撞點位置列表
        target_pos: 當前目標 (x, y)
        local_waypoints: 即即将到達的路徑點列表
    """
    # Draw path to target
    if len(local_waypoints) > 0:
        ref_x = car_physics.nose_c_x
        ref_y = car_physics.nose_c_y
        path_points = [
            (int(ref_x), int(ref_y)), 
            (int(target_pos[0]), int(target_pos[1]))
        ] + [(int(wp[0]), int(wp[1])) for wp in local_waypoints]
        pygame.draw.lines(window, (0, 150, 255), False, path_points, 2)
        for wp in local_waypoints:
            pygame.draw.circle(window, (0, 150, 255), (int(wp[0]), int(wp[1])), 3)
    else:
        ref_x = car_physics.nose_c_x
        ref_y = car_physics.nose_c_y
        pygame.draw.line(window, (0, 150, 255), (int(ref_x), int(ref_y)), 
                        (int(target_pos[0]), int(target_pos[1])), 2)

    # Draw target
    pygame.draw.circle(window, (255,0,0), (int(target_pos[0]), int(target_pos[1])), 6)
        
    # Draw LIDAR rays
    for hit in lidar_hits:
        pygame.draw.line(window, (0, 200, 255, 100), (int(car_physics.car_x), int(car_physics.car_y)), 
                        (int(hit[0]), int(hit[1])), 1)
        pygame.draw.circle(window, (255, 100, 0), (int(hit[0]), int(hit[1])), 2)
            
    # Draw car body
    car_s = pygame.Surface((car_physics.car_w, car_physics.car_h), pygame.SRCALPHA)
    car_s.fill((0,0,255))
    pygame.draw.rect(car_s, (0,255,255), (0, 0, car_physics.car_w, 10))
    rot_car = pygame.transform.rotate(car_s, -car_physics.car_angle)
    window.blit(rot_car, rot_car.get_rect(center=(int(car_physics.car_x), int(car_physics.car_y))))
    
    # Draw front bumpers (yellow)
    pygame.draw.circle(window, (255, 255, 0), (int(car_physics.nose_l_x), int(car_physics.nose_l_y)), 4)
    pygame.draw.circle(window, (255, 255, 0), (int(car_physics.nose_c_x), int(car_physics.nose_c_y)), 4)
    pygame.draw.circle(window, (255, 255, 0), (int(car_physics.nose_r_x), int(car_physics.nose_r_y)), 4)
    
    # Draw rear bumpers (green)
    pygame.draw.circle(window, (0, 255, 0), (int(car_physics.rear_l_x), int(car_physics.rear_l_y)), 4)
    pygame.draw.circle(window, (0, 255, 0), (int(car_physics.rear_c_x), int(car_physics.rear_c_y)), 4)
    pygame.draw.circle(window, (0, 255, 0), (int(car_physics.rear_r_x), int(car_physics.rear_r_y)), 4)
