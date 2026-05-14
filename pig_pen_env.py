import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import heapq

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

        if nx < self.min_x and vx < 0: hit_something = True
        if nx > self.max_x and vx > 0: hit_something = True
        if ny < self.min_y and vy < 0: hit_something = True
        if ny > self.max_y and vy > 0: hit_something = True

        if not hit_something:
            dist_to_bin = math.hypot(nx - self.bin_pos[0], ny - self.bin_pos[1])
            curr_dist_to_bin = math.hypot(self.x - self.bin_pos[0], self.y - self.bin_pos[1])
            if dist_to_bin < (self.bin_r + self.collision_radius) and dist_to_bin < curr_dist_to_bin:
                hit_something = True

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
        pig_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pig_surf.fill((255, 182, 193)) 
        nose_h = max(4, int(15 * self.scale))
        pygame.draw.rect(pig_surf, (255, 105, 180), (0, 0, self.w, nose_h)) 
        rotated = pygame.transform.rotate(pig_surf, -self.angle)
        surface.blit(rotated, rotated.get_rect(center=(int(self.x), int(self.y))))

# ==========================================
# 強化學習環境主體 (A* + RL 混合架構)
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
            pygame.display.set_caption("Hierarchical Nav: Full Bumper Sensors")
            self.clock = pygame.time.Clock()
        else:
            pygame.init()
            self.window = pygame.Surface((self.screen_w, self.screen_h))
            
        self.obstacle_surface = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        self.pen_info = [] 
        
        self._draw_static_obstacles()
        self._build_grid_map()
        self.global_goals = self._generate_global_goals()
        self.pigs = []

    def _generate_global_goals(self):
        goals = []
        cl, cr = self.center_x - self.corridor_w/2, self.center_x + self.corridor_w/2
        for i in range(3, -1, -1):
            y = i * (self.pen_h + 5) + 20 + self.pen_h/2
            goals.append((cl - self.pen_w/4, y))
            goals.append((cr + self.pen_w/4, y))
        return goals

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
            
            self.pen_info.append({"bounds": (cl-self.pen_w, y, cl, y+self.pen_h), "bin_pos": (cl-self.pen_w/2, y+self.pen_h/2)})
            pygame.draw.line(self.obstacle_surface, wall_color, (cl - self.pen_w, y), (cl, y), wall_thick)
            pygame.draw.line(self.obstacle_surface, wall_color, (cl - self.pen_w, y), (cl - self.pen_w, y + self.pen_h), wall_thick)
            pygame.draw.line(self.obstacle_surface, wall_color, (cl - self.pen_w, y + self.pen_h), (cl, y + self.pen_h), wall_thick)
            pygame.draw.line(self.obstacle_surface, wall_color, (cl, y), (cl, dy), wall_thick)
            pygame.draw.line(self.obstacle_surface, wall_color, (cl, dy + self.door_w), (cl, y + self.pen_h), wall_thick)
            pygame.draw.circle(self.obstacle_surface, (0, 255, 0, 255), (cl - self.pen_w/2, y + self.pen_h/2), self.bin_r)

            self.pen_info.append({"bounds": (cr, y, cr+self.pen_w, y+self.pen_h), "bin_pos": (cr+self.pen_w/2, y+self.pen_h/2)})
            pygame.draw.line(self.obstacle_surface, wall_color, (cr, y), (cr + self.pen_w, y), wall_thick)
            pygame.draw.line(self.obstacle_surface, wall_color, (cr + self.pen_w, y), (cr + self.pen_w, y + self.pen_h), wall_thick)
            pygame.draw.line(self.obstacle_surface, wall_color, (cr, y + self.pen_h), (cr + self.pen_w, y + self.pen_h), wall_thick)
            pygame.draw.line(self.obstacle_surface, wall_color, (cr, y), (cr, dy), wall_thick)
            pygame.draw.line(self.obstacle_surface, wall_color, (cr, dy + self.door_w), (cr, y + self.pen_h), wall_thick)
            pygame.draw.circle(self.obstacle_surface, (0, 255, 0, 255), (cr + self.pen_w/2, y + self.pen_h/2), self.bin_r)

    def _build_grid_map(self):
        self.cell_size = 5 
        self.cols = self.screen_w // self.cell_size
        self.rows = self.screen_h // self.cell_size
        self.grid = np.zeros((self.cols, self.rows), dtype=int)
        
        inflation_radius = int(max(self.car_w, self.car_h) / 2)
        scan_step = 2 
        
        for cx in range(self.cols):
            for cy in range(self.rows):
                px = cx * self.cell_size + self.cell_size // 2
                py = cy * self.cell_size + self.cell_size // 2
                
                is_obs = False
                for dx in range(-inflation_radius, inflation_radius+1, scan_step):
                    for dy in range(-inflation_radius, inflation_radius+1, scan_step):
                        check_x, check_y = px + dx, py + dy
                        if 0 <= check_x < self.screen_w and 0 <= check_y < self.screen_h:
                            if self.obstacle_surface.get_at((int(check_x), int(check_y)))[3] > 0:
                                is_obs = True
                                break
                    if is_obs: break
                self.grid[cx, cy] = 1 if is_obs else 0

        self.debug_grid_surface = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        for cx in range(self.cols):
            for cy in range(self.rows):
                if self.grid[cx, cy] == 1:
                    pygame.draw.rect(self.debug_grid_surface, (255, 0, 0, 30), 
                                     (cx*self.cell_size, cy*self.cell_size, self.cell_size, self.cell_size))

    def _get_nearest_free_cell(self, start_c):
        if 0 <= start_c[0] < self.cols and 0 <= start_c[1] < self.rows:
            if self.grid[start_c[0], start_c[1]] == 0: return start_c
            
        queue = [start_c]
        visited = set([start_c])
        while queue:
            curr = queue.pop(0)
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.grid[nx, ny] == 0: return (nx, ny)
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return start_c

    def _a_star(self, start_pos, goal_pos):
        start_c = (int(start_pos[0] // self.cell_size), int(start_pos[1] // self.cell_size))
        goal_c = (int(goal_pos[0] // self.cell_size), int(goal_pos[1] // self.cell_size))
        
        start_c = self._get_nearest_free_cell(start_c)
        goal_c = self._get_nearest_free_cell(goal_c)
        
        if not (0 <= start_c[0] < self.cols and 0 <= start_c[1] < self.rows): return []
            
        frontier = []
        heapq.heappush(frontier, (0, start_c))
        came_from = {start_c: None}
        cost_so_far = {start_c: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_c: break
            
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                next_c = (current[0] + dx, current[1] + dy)
                if 0 <= next_c[0] < self.cols and 0 <= next_c[1] < self.rows:
                    if self.grid[next_c] == 1: continue 
                    
                    if dx != 0 and dy != 0:
                        if self.grid[current[0]+dx, current[1]] == 1 or self.grid[current[0], current[1]+dy] == 1:
                            continue
                    
                    move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    new_cost = cost_so_far[current] + move_cost
                    
                    if next_c not in cost_so_far or new_cost < cost_so_far[next_c]:
                        cost_so_far[next_c] = new_cost
                        priority = new_cost + math.hypot(goal_c[0] - next_c[0], goal_c[1] - next_c[1])
                        heapq.heappush(frontier, (priority, next_c))
                        came_from[next_c] = current
        
        if goal_c not in came_from: return [goal_pos]
            
        path = []
        curr = goal_c
        while curr != start_c:
            path.append((curr[0] * self.cell_size + self.cell_size//2, curr[1] * self.cell_size + self.cell_size//2))
            curr = came_from[curr]
        path.reverse()
        
        sampled_path = path[::15]
        if goal_pos not in sampled_path:
            sampled_path.append(goal_pos)
            
        return sampled_path

    # ==========================================
    # 🌟 更新所有保險桿感測器座標的函數 (前後各三個)
    # ==========================================
    def _update_car_points(self):
        rad = math.radians(self.car_angle)
        fw_x = math.sin(rad)
        fw_y = -math.cos(rad)
        rt_x = math.cos(rad)
        rt_y = math.sin(rad)

        # 車頭正中央 (Yellow Center)
        self.nose_c_x = self.car_x + (self.car_h / 2) * fw_x
        self.nose_c_y = self.car_y + (self.car_h / 2) * fw_y

        # 車頭右側 (Yellow Right) - 縮2像素避免視覺超出藍色車身
        self.nose_r_x = self.nose_c_x + (self.car_w / 2 - 2) * rt_x
        self.nose_r_y = self.nose_c_y + (self.car_w / 2 - 2) * rt_y

        # 車頭左側 (Yellow Left)
        self.nose_l_x = self.nose_c_x - (self.car_w / 2 - 2) * rt_x
        self.nose_l_y = self.nose_c_y - (self.car_w / 2 - 2) * rt_y

        # 車尾正中央 (Green Rear Center)
        self.rear_c_x = self.car_x - (self.car_h / 2) * fw_x
        self.rear_c_y = self.car_y - (self.car_h / 2) * fw_y
        
        # 車尾右側 (Green Rear Right)
        self.rear_r_x = self.rear_c_x + (self.car_w / 2 - 2) * rt_x
        self.rear_r_y = self.rear_c_y + (self.car_w / 2 - 2) * rt_y
        
        # 車尾左側 (Green Rear Left)
        self.rear_l_x = self.rear_c_x - (self.car_w / 2 - 2) * rt_x
        self.rear_l_y = self.rear_c_y - (self.car_w / 2 - 2) * rt_y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_x, self.car_y, self.car_angle, self.steps = self.center_x, 750.0, 0.0, 0
        
        self._update_car_points()
        
        self.current_goal_idx = 0
        self.local_waypoints = self._a_star((self.car_x, self.car_y), self.global_goals[self.current_goal_idx])
        
        if len(self.local_waypoints) > 0:
            self.target_pos = self.local_waypoints.pop(0)
        else:
            self.target_pos = self.global_goals[self.current_goal_idx]
            
        is_tracking_final = (len(self.local_waypoints) == 0)
        # 一般追蹤距離仍然以「車頭中心」或「車體中心」為基準
        ref_x = self.nose_c_x if is_tracking_final else self.car_x
        ref_y = self.nose_c_y if is_tracking_final else self.car_y
        self.prev_dist = math.hypot(ref_x - self.target_pos[0], ref_y - self.target_pos[1])
        
        self.pigs = []
        pid = 0
        for info in self.pen_info:
            bx, by = info["bin_pos"]
            safe_spawn_points = [(bx - 35, by - 40), (bx + 35, by + 40)]
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

        # 即時更新所有感測器的位置
        self._update_car_points()

        car_s = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        car_s.fill((0,0,255,255))
        rot_car = pygame.transform.rotate(car_s, -self.car_angle)
        rect = rot_car.get_rect(center=(int(self.car_x), int(self.car_y)))
        is_collision = self.dynamic_mask.overlap(pygame.mask.from_surface(rot_car), (rect.x, rect.y)) is not None

        is_tracking_final = (len(self.local_waypoints) == 0)
        
        ref_x = self.nose_c_x if is_tracking_final else self.car_x
        ref_y = self.nose_c_y if is_tracking_final else self.car_y

        dist = math.hypot(ref_x - self.target_pos[0], ref_y - self.target_pos[1])
        trigger_dist = 15.0 if is_tracking_final else 30.0

        # 計算三個前保險桿(黃點) 到最終紅點的距離
        dist_n_c = math.hypot(self.nose_c_x - self.target_pos[0], self.nose_c_y - self.target_pos[1])
        dist_n_l = math.hypot(self.nose_l_x - self.target_pos[0], self.nose_l_y - self.target_pos[1])
        dist_n_r = math.hypot(self.nose_r_x - self.target_pos[0], self.nose_r_y - self.target_pos[1])
        min_front_dist = min(dist_n_c, dist_n_l, dist_n_r) # 只要有一個碰到就算數

        # 🌟 計算三個後保險桿(綠點) 到最終紅點的距離
        dist_r_c = math.hypot(self.rear_c_x - self.target_pos[0], self.rear_c_y - self.target_pos[1])
        dist_r_l = math.hypot(self.rear_l_x - self.target_pos[0], self.rear_l_y - self.target_pos[1])
        dist_r_r = math.hypot(self.rear_r_x - self.target_pos[0], self.rear_r_y - self.target_pos[1])
        min_rear_dist = min(dist_r_c, dist_r_l, dist_r_r) # 只要有一個碰到就算數

        target_ang = math.degrees(math.atan2(self.target_pos[0]-ref_x, -(self.target_pos[1]-ref_y)))
        rel_ang_deg = (target_ang - self.car_angle + 180) % 360 - 180
        rel_ang_norm = abs(rel_ang_deg / 180.0) 

        current_global_goal = self.global_goals[self.current_goal_idx]
        dist_to_global = math.hypot(self.car_x - current_global_goal[0], self.car_y - current_global_goal[1])

        if dist_to_global < 150.0:
            heading_penalty = rel_ang_norm * 0.05  
        else:
            heading_penalty = 0.0  

        time_penalty = 0.02
        reward = -time_penalty - heading_penalty + (self.prev_dist - dist) * 0.5
        self.prev_dist = dist

        terminated = False
        
        # 全新的碰撞與吃點邏輯
        if is_collision: 
            reward -= 50.0
            terminated = True
            
        elif is_tracking_final and min_rear_dist < trigger_dist:
            # 🚨 致命錯誤：車尾任何一個綠點觸碰到了飼料桶紅點！
            # 給予跟撞牆一樣的重罰，並直接結束回合
            reward -= 50.0
            terminated = True
            print("🚨 警告：車尾觸碰目標！判定失敗！")
            
        elif (not is_tracking_final and dist < trigger_dist) or (is_tracking_final and min_front_dist < trigger_dist):
            # 成功吃到點！
            if not is_tracking_final:
                self.target_pos = self.local_waypoints.pop(0)
                reward += 1.0  
            else:
                room_bonus = 20.0 * (1.0 - rel_ang_norm)
                reward += room_bonus

                self.current_goal_idx += 1
                if self.current_goal_idx >= len(self.global_goals): 
                    reward += 1000.0
                    terminated = True
                else:
                    self.local_waypoints = self._a_star((self.car_x, self.car_y), self.global_goals[self.current_goal_idx])
                    if len(self.local_waypoints) > 0:
                        self.target_pos = self.local_waypoints.pop(0)
                    else:
                        self.target_pos = self.global_goals[self.current_goal_idx]
            
            is_tracking_final_new = (len(self.local_waypoints) == 0)
            new_ref_x = self.nose_c_x if is_tracking_final_new else self.car_x
            new_ref_y = self.nose_c_y if is_tracking_final_new else self.car_y
            self.prev_dist = math.hypot(new_ref_x - self.target_pos[0], new_ref_y - self.target_pos[1])

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
        
        is_tracking_final = (len(self.local_waypoints) == 0)
        ref_x = self.nose_c_x if is_tracking_final else self.car_x
        ref_y = self.nose_c_y if is_tracking_final else self.car_y

        rel_dist = min(math.hypot(self.target_pos[0]-ref_x, self.target_pos[1]-ref_y)/self.screen_h, 1.0)
        target_ang = math.degrees(math.atan2(self.target_pos[0]-ref_x, -(self.target_pos[1]-ref_y)))
        rel_ang = ((target_ang - self.car_angle + 180) % 360 - 180) / 180.0
        return np.array(lidar + [rel_dist, rel_ang], dtype=np.float32)

    def render(self):
        if self.render_mode != "human": return
        self.window.fill((255,255,255))
        self.window.blit(self.dynamic_surface, (0,0))
        
        self.window.blit(self.debug_grid_surface, (0,0)) 
        
        is_tracking_final = (len(self.local_waypoints) == 0)
        ref_x = self.nose_c_x if is_tracking_final else self.car_x
        ref_y = self.nose_c_y if is_tracking_final else self.car_y

        if len(self.local_waypoints) > 0:
            path_points = [
                (int(ref_x), int(ref_y)), 
                (int(self.target_pos[0]), int(self.target_pos[1]))
            ] + [(int(wp[0]), int(wp[1])) for wp in self.local_waypoints]
            
            pygame.draw.lines(self.window, (0, 150, 255), False, path_points, 2)
            for wp in self.local_waypoints:
                pygame.draw.circle(self.window, (0, 150, 255), (int(wp[0]), int(wp[1])), 3)
        else:
            pygame.draw.line(self.window, (0, 150, 255), (int(ref_x), int(ref_y)), (int(self.target_pos[0]), int(self.target_pos[1])), 2)

        pygame.draw.circle(self.window, (255,0,0), (int(self.target_pos[0]), int(self.target_pos[1])), 6)
            
        for hit in self.lidar_hits:
            pygame.draw.line(self.window, (0, 200, 255, 100), (int(self.car_x), int(self.car_y)), (int(hit[0]), int(hit[1])), 1)
            pygame.draw.circle(self.window, (255, 100, 0), (int(hit[0]), int(hit[1])), 2)
            
        car_s = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        car_s.fill((0,0,255))
        pygame.draw.rect(car_s, (0,255,255), (0,0,self.car_w, 10))
        rot_car = pygame.transform.rotate(car_s, -self.car_angle)
        self.window.blit(rot_car, rot_car.get_rect(center=(int(self.car_x), int(self.car_y))))
        
        # ==========================================
        # 🌟 視覺化感測器
        # ==========================================
        # 畫出 3 個黃色前保險桿感測器
        pygame.draw.circle(self.window, (255, 255, 0), (int(self.nose_l_x), int(self.nose_l_y)), 4)
        pygame.draw.circle(self.window, (255, 255, 0), (int(self.nose_c_x), int(self.nose_c_y)), 4)
        pygame.draw.circle(self.window, (255, 255, 0), (int(self.nose_r_x), int(self.nose_r_y)), 4)
        
        # 🌟 畫出 3 個綠色後防撞感測器
        pygame.draw.circle(self.window, (0, 255, 0), (int(self.rear_l_x), int(self.rear_l_y)), 4)
        pygame.draw.circle(self.window, (0, 255, 0), (int(self.rear_c_x), int(self.rear_c_y)), 4)
        pygame.draw.circle(self.window, (0, 255, 0), (int(self.rear_r_x), int(self.rear_r_y)), 4)
        
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