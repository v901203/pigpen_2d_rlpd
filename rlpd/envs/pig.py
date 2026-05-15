"""
豬隻動力學與行為模擬模組

此模組定義 Pig 類別，模擬豬舍中動態豬隻的運動、碰撞與行為。
豬隻會隨機遊走、與牆壁/飼料桶/其他豬隻發生碰撞反彈。
"""
import numpy as np
import pygame
import math


class Pig:
    """Represents a dynamic pig in the environment."""
    
    def __init__(self, spawn_pos, bounds, bin_pos, bin_r, scale, id):
        """
        初始化豬隻物體
        
        參數：
            spawn_pos: (x, y) 生成位置
            bounds: (x_min, y_min, x_max, y_max) 環境邊界
            bin_pos: (x, y) 飼料桶位置
            bin_r: 飼料桶半徑
            scale: 視覺縮放係數
            id: 豬隻唯一識別碼
        """
        self.id = id 
        self.bounds = bounds 
        self.bin_pos = bin_pos
        self.bin_r = bin_r
        self.scale = scale
        
        self.w = 45 * scale
        self.h = 120 * scale 
        self.collision_radius = (self.h / 2) * 0.8 
        
        self.x, self.y = spawn_pos
        self.angle = np.random.uniform(0, 360)
        self.v = 0.0
        self.omega = 0.0
        self.action_timer = 0
        
        margin = self.h / 2 + 5
        self.min_x, self.max_x = bounds[0] + margin, bounds[2] - margin
        self.min_y, self.max_y = bounds[1] + margin, bounds[3] - margin

    def update(self, other_pigs):
        """更新豬隻位置與行為（定時隨機決策與碰撞反應）"""
        self.action_timer -= 1
        if self.action_timer <= 0:
            action_type = np.random.choice(['walk', 'turn', 'stop'], p=[0.2, 0.3, 0.5])
            if action_type == 'walk':
                self.v = np.random.uniform(0.2, 0.6) 
                self.omega = np.random.uniform(-0.5, 0.5)
            elif action_type == 'turn':
                self.v = 0.0
                self.omega = np.random.uniform(-2.0, 2.0)
            else:
                self.v, self.omega = 0.0, 0.0
            self.action_timer = np.random.randint(100, 200) 

        next_angle = (self.angle + self.omega) % 360
        rad = math.radians(next_angle)
        vx = self.v * math.sin(rad)
        vy = -self.v * math.cos(rad)
        nx = self.x + vx
        ny = self.y + vy

        hit_something = False

        # Boundary collision check
        if nx < self.min_x and vx < 0: hit_something = True
        if nx > self.max_x and vx > 0: hit_something = True
        if ny < self.min_y and vy < 0: hit_something = True
        if ny > self.max_y and vy > 0: hit_something = True

        # Bin collision check
        if not hit_something:
            dist_to_bin = math.hypot(nx - self.bin_pos[0], ny - self.bin_pos[1])
            curr_dist_to_bin = math.hypot(self.x - self.bin_pos[0], self.y - self.bin_pos[1])
            if dist_to_bin < (self.bin_r + self.collision_radius) and dist_to_bin < curr_dist_to_bin:
                hit_something = True

        # Pig-to-pig collision check
        if not hit_something:
            for other in other_pigs:
                if other.id == self.id: continue
                dist = math.hypot(nx - other.x, ny - other.y)
                curr_dist = math.hypot(self.x - other.x, self.y - other.y)
                min_dist = self.collision_radius + other.collision_radius
                
                if dist < min_dist and dist < curr_dist:
                    hit_something = True
                    break

        if hit_something:
            self.angle = (self.angle + 180 + np.random.uniform(-45, 45)) % 360
            self.v = 0 
            self.action_timer = 30 
        else:
            self.x = nx
            self.y = ny
            self.angle = next_angle

    def draw(self, surface):
        """在 pygame 表面繪製豬隻"""
        pig_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pig_surf.fill((255, 182, 193))  # Light pink body
        nose_h = max(4, int(15 * self.scale))
        pygame.draw.rect(pig_surf, (255, 105, 180), (0, 0, self.w, nose_h))  # Hot pink snout
        rotated = pygame.transform.rotate(pig_surf, -self.angle)
        surface.blit(rotated, rotated.get_rect(center=(int(self.x), int(self.y))))
