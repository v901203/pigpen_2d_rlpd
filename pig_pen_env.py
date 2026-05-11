import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

# ==========================================
# 定義動態豬隻類別 (單向物理阻擋：防抽搐卡死)
# ==========================================
class Pig:
    def __init__(self, spawn_pos, bounds, bin_pos, bin_r, scale, id):
        self.id = id 
        self.bounds = bounds 
        self.bin_pos = bin_pos
        self.bin_r = bin_r
        self.scale = scale
        
        # 成年肉豬尺寸 (約 120cm x 45cm)
        self.w = 45 * scale
        self.h = 120 * scale 
        
        # 實體碰撞半徑
        self.collision_radius = (self.h / 2) * 0.8 
        
        self.x, self.y = spawn_pos
        self.angle = np.random.uniform(0, 360)
        self.v = 0.0
        self.omega = 0.0
        self.action_timer = 0
        
        # 活動邊界
        margin = self.h / 2 + 5
        self.min_x, self.max_x = bounds[0] + margin, bounds[2] - margin
        self.min_y, self.max_y = bounds[1] + margin, bounds[3] - margin

    def update(self, other_pigs):
        self.action_timer -= 1
        if self.action_timer <= 0:
            # 50% 停止，30% 原地轉向，20% 慢走
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

        # 🌟 1. 牆壁防卡死：只有「下一步比現在更往外出界」時才擋住
        if nx < self.min_x and vx < 0: hit_something = True
        if nx > self.max_x and vx > 0: hit_something = True
        if ny < self.min_y and vy < 0: hit_something = True
        if ny > self.max_y and vy > 0: hit_something = True

        # 🌟 2. 飼料桶防卡死：只有「下一步比現在更靠近飼料桶」時才擋住
        if not hit_something:
            dist_to_bin = math.hypot(nx - self.bin_pos[0], ny - self.bin_pos[1])
            curr_dist_to_bin = math.hypot(self.x - self.bin_pos[0], self.y - self.bin_pos[1])
            if dist_to_bin < (self.bin_r + self.collision_radius) and dist_to_bin < curr_dist_to_bin:
                hit_something = True

        # 🌟 3. 豬隻間防卡死：只有「下一步比現在更靠近其他豬」時才擋住
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
            # 撞到東西，強制轉向並停下思考
            self.angle = (self.angle + 180 + np.random.uniform(-45, 45)) % 360
            self.v = 0 
            self.action_timer = 30 
        else:
            self.x = nx
            self.y = ny
            self.angle = next_angle

    def draw(self, surface):
        pig_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pig_surf.fill((255, 182, 193)) 
        nose_h = max(4, int(15 * self.scale))
        pygame.draw.rect(pig_surf, (255, 105, 180), (0, 0, self.w, nose_h)) 
        rotated = pygame.transform.rotate(pig_surf, -self.angle)
        surface.blit(rotated, rotated.get_rect(center=(int(self.x), int(self.y))))

# ==========================================
# 強化學習環境主體
# ==========================================
class PigPenEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, door_width_scale=1.0):
        super().__init__()
        self.render_mode = render_mode
        self.scale = 0.6 
        self.car_w, self.car_h = 40 * self.scale, 45 * self.scale
        self.door_w = 59 * self.scale * door_width_scale
        self.pen_w, self.pen_h = 275 * self.scale, 289 * self.scale
        self.corridor_w = 150 * self.scale
        self.bin_r = (60 / 2) * self.scale 
        self.screen_w, self.screen_h = 600, 800
        self.center_x = self.screen_w // 2
        self.num_rays = 36          
        self.lidar_range = 250.0    
        self.lidar_hits = [] 

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(38,), dtype=np.float32)

        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("RL Env: Fixed Pig Physics")
            self.clock = pygame.time.Clock()
        else:
            pygame.init()
            self.window = pygame.Surface((self.screen_w, self.screen_h))
            
        self.obstacle_surface = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        self.pen_info = [] 
        self._draw_static_obstacles()
        self.waypoints = self._generate_waypoints()
        self.pigs = []

    def _generate_waypoints(self):
        wps = []
        cl, cr = self.center_x - self.corridor_w/2, self.center_x + self.corridor_w/2
        for i in range(3, -1, -1):
            y = i * (self.pen_h + 5) + 20 + self.pen_h/2
            wps.append((cl - self.pen_w/4, y))
            wps.append((cr + self.pen_w/4, y))
        return wps

    def _draw_static_obstacles(self):
        self.obstacle_surface.fill((0, 0, 0, 0))
        pygame.draw.rect(self.obstacle_surface, (0,0,0,255), (0, 0, self.screen_w, self.screen_h), 5)
        
        cl = self.center_x - (self.corridor_w / 2)
        cr = self.center_x + (self.corridor_w / 2)
        wall_color = (0, 0, 0, 255)
        wall_thick = 2

        for i in range(4):
            y = i * (self.pen_h + 5) + 20
            dy = y + self.pen_h / 2 - self.door_w / 2
            
            # --- 左側房間 ---
            self.pen_info.append({"bounds": (cl-self.pen_w, y, cl, y+self.pen_h), "bin_pos": (cl-self.pen_w/2, y+self.pen_h/2)})
            pygame.draw.line(self.obstacle_surface, wall_color, (cl - self.pen_w, y), (cl, y), wall_thick) # 上
            pygame.draw.line(self.obstacle_surface, wall_color, (cl - self.pen_w, y), (cl - self.pen_w, y + self.pen_h), wall_thick) # 左
            pygame.draw.line(self.obstacle_surface, wall_color, (cl - self.pen_w, y + self.pen_h), (cl, y + self.pen_h), wall_thick) # 下
            pygame.draw.line(self.obstacle_surface, wall_color, (cl, y), (cl, dy), wall_thick) # 右上門框
            pygame.draw.line(self.obstacle_surface, wall_color, (cl, dy + self.door_w), (cl, y + self.pen_h), wall_thick) # 右下門框
            pygame.draw.circle(self.obstacle_surface, (0, 255, 0, 255), (cl - self.pen_w/2, y + self.pen_h/2), self.bin_r)

            # --- 右側房間 ---
            self.pen_info.append({"bounds": (cr, y, cr+self.pen_w, y+self.pen_h), "bin_pos": (cr+self.pen_w/2, y+self.pen_h/2)})
            pygame.draw.line(self.obstacle_surface, wall_color, (cr, y), (cr + self.pen_w, y), wall_thick) # 上
            pygame.draw.line(self.obstacle_surface, wall_color, (cr + self.pen_w, y), (cr + self.pen_w, y + self.pen_h), wall_thick) # 右
            pygame.draw.line(self.obstacle_surface, wall_color, (cr, y + self.pen_h), (cr + self.pen_w, y + self.pen_h), wall_thick) # 下
            pygame.draw.line(self.obstacle_surface, wall_color, (cr, y), (cr, dy), wall_thick) # 左上門框
            pygame.draw.line(self.obstacle_surface, wall_color, (cr, dy + self.door_w), (cr, y + self.pen_h), wall_thick) # 左下門框
            pygame.draw.circle(self.obstacle_surface, (0, 255, 0, 255), (cr + self.pen_w/2, y + self.pen_h/2), self.bin_r)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_x, self.car_y, self.car_angle, self.steps = self.center_x, 750.0, 0.0, 0
        self.current_wp_idx = 0
        self.target_pos = self.waypoints[0]
        self.prev_dist = math.hypot(self.car_x - self.target_pos[0], self.car_y - self.target_pos[1])
        
        self.pigs = []
        pid = 0
        for info in self.pen_info:
            bx, by = info["bin_pos"]
            # 🌟 修正：精確計算安全重生點，保證絕對不會生成在牆壁或飼料桶內
            safe_spawn_points = [
                (bx - 35, by - 40), 
                (bx + 35, by + 40)
            ]
            for pos in safe_spawn_points:
                self.pigs.append(Pig(pos, info["bounds"], info["bin_pos"], self.bin_r, self.scale, pid))
                pid += 1
        self._update_dynamic()
        return self._get_obs(), {}

    def _update_dynamic(self):
        self.dynamic_surface = self.obstacle_surface.copy()
        for p in self.pigs: p.draw(self.dynamic_surface)
        self.dynamic_mask = pygame.mask.from_surface(self.dynamic_surface)

    def step(self, action):
        self.steps += 1
        for p in self.pigs: p.update(self.pigs)
        self._update_dynamic()
        
        v, omega = np.clip(action[0],-1,1)*5.0, np.clip(action[1],-1,1)*5.0
        self.car_angle += omega
        rad = math.radians(self.car_angle)
        self.car_x += v * math.sin(rad)
        self.car_y -= v * math.cos(rad)

        car_s = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        car_s.fill((0,0,255,255))
        rot_car = pygame.transform.rotate(car_s, -self.car_angle)
        rect = rot_car.get_rect(center=(int(self.car_x), int(self.car_y)))
        is_collision = self.dynamic_mask.overlap(pygame.mask.from_surface(rot_car), (rect.x, rect.y)) is not None

        dist = math.hypot(self.car_x - self.target_pos[0], self.car_y - self.target_pos[1])
        
        # 🌟 關鍵修改：降低時間懲罰 (從 0.1 降為 0.02)
        # 這樣小車就不會因為急著趕路而直接衝去撞豬「自殺」了
        time_penalty = 0.02
        reward = -time_penalty + (self.prev_dist - dist) * 0.5
        
        self.prev_dist = dist
        
        terminated = False
        if is_collision: 
            reward -= 50.0
            terminated = True
        elif dist < 20.0:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints): 
                reward += 1000.0
                terminated = True
            else:
                reward += 50.0 + (self.current_wp_idx * 20.0)
                self.target_pos = self.waypoints[self.current_wp_idx]
                self.prev_dist = math.hypot(self.car_x - self.target_pos[0], self.car_y - self.target_pos[1])

        if self.render_mode == "human": self.render()
        return self._get_obs(), float(reward), terminated, self.steps>=5000, {}

    def _get_obs(self):
        lidar = []
        self.lidar_hits = [] 
        
        for i in range(self.num_rays):
            rad = math.radians(self.car_angle + i*(360/self.num_rays))
            dx, dy = math.sin(rad), -math.cos(rad)
            d_res = self.lidar_range
            hit_pos = (self.car_x + dx * self.lidar_range, self.car_y + dy * self.lidar_range)
            
            for d in range(0, int(self.lidar_range), 5):
                px, py = int(self.car_x + dx*d), int(self.car_y + dy*d)
                if not (0<=px<self.screen_w and 0<=py<self.screen_h) or self.dynamic_mask.get_at((px,py)):
                    d_res = d
                    hit_pos = (px, py)
                    break
                    
            lidar.append(d_res / self.lidar_range)
            self.lidar_hits.append(hit_pos)
        
        rel_dist = min(math.hypot(self.target_pos[0]-self.car_x, self.target_pos[1]-self.car_y)/self.screen_h, 1.0)
        target_ang = math.degrees(math.atan2(self.target_pos[0]-self.car_x, -(self.target_pos[1]-self.car_y)))
        rel_ang = ((target_ang - self.car_angle + 180) % 360 - 180) / 180.0
        return np.array(lidar + [rel_dist, rel_ang], dtype=np.float32)

    def render(self):
        if self.render_mode != "human": return
        self.window.fill((255,255,255))
        self.window.blit(self.dynamic_surface, (0,0))
        
        for i, wp in enumerate(self.waypoints):
            pygame.draw.circle(self.window, (255,0,0) if i==self.current_wp_idx else (200,200,200), (int(wp[0]), int(wp[1])), 6 if i==self.current_wp_idx else 3)
            
        for hit in self.lidar_hits:
            pygame.draw.line(self.window, (0, 200, 255, 100), (int(self.car_x), int(self.car_y)), (int(hit[0]), int(hit[1])), 1)
            pygame.draw.circle(self.window, (255, 100, 0), (int(hit[0]), int(hit[1])), 2)
            
        car_s = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        car_s.fill((0,0,255))
        pygame.draw.rect(car_s, (0,255,255), (0,0,self.car_w, 10))
        rot_car = pygame.transform.rotate(car_s, -self.car_angle)
        self.window.blit(rot_car, rot_car.get_rect(center=(int(self.car_x), int(self.car_y))))
        pygame.display.flip()
        self.clock.tick(30)

if __name__ == "__main__":
    env = PigPenEnv(render_mode="human")
    obs, _ = env.reset()
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
        if done or trunc: env.reset()