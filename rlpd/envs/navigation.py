"""
路徑規劃與導航模組

此模組提供：
1. 網格地圖構建：根據障礙物表面生成碰撞網格
2. A* 尋路演算法：計算从當前位置到目標的最優路徑
3. 全局目標生成：自動創建 8 個豬舍房間的目標點
4. 障礙物繪製：在 pygame 表面繪製牆壁、門、飼料桶
"""
import numpy as np
import pygame
import math
import heapq


class NavigationGrid:
    """Grid-based navigation and pathfinding."""
    
    def __init__(self, screen_w, screen_h, cell_size=5):
        """
        初始化導航網格
        
        參數：
            screen_w: 螢幕寬度（像素）
            screen_h: 螢幕高度（像素）
            cell_size: 網格單元大小（像素，預設 5 像素 = 1 單元）
        """
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cell_size = cell_size
        self.cols = screen_w // cell_size
        self.rows = screen_h // cell_size
        self.grid = np.zeros((self.cols, self.rows), dtype=int)
        self.debug_grid_surface = None

    def build_from_obstacle_surface(self, obstacle_surface, car_w, car_h):
        """
        從障礙物表面構建碰撞網格
        
        原理：掃描每個網格單元，若半徑內有障礙物像素則標記為碰撞格
        膨脹效果：根據車體大小膨脹障礙物，避免路徑太靠近牆壁
        
        參數：
            obstacle_surface: 繪有障礙物的 pygame 表面
            car_w: 車寬（用於計算膨脹半徑）
            car_h: 車高（用於計算膨脹半徑）
        """
        inflation_radius = int(max(car_w, car_h) / 2)
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
                            if obstacle_surface.get_at((int(check_x), int(check_y)))[3] > 0:
                                is_obs = True
                                break
                    if is_obs: break
                self.grid[cx, cy] = 1 if is_obs else 0

        # Create debug visualization
        self.debug_grid_surface = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        for cx in range(self.cols):
            for cy in range(self.rows):
                if self.grid[cx, cy] == 1:
                    pygame.draw.rect(self.debug_grid_surface, (255, 0, 0, 30), 
                                     (cx*self.cell_size, cy*self.cell_size, self.cell_size, self.cell_size))

    def get_nearest_free_cell(self, start_c):
        """使用廣度優先搜尋（BFS）找到最近的自由格子
        
        用途：當起點或終點恰好在障礙物上時，向外擴展尋找最近的可走格子
        """
        if 0 <= start_c[0] < self.cols and 0 <= start_c[1] < self.rows:
            if self.grid[start_c[0], start_c[1]] == 0: 
                return start_c
            
        queue = [start_c]
        visited = set([start_c])
        while queue:
            curr = queue.pop(0)
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.grid[nx, ny] == 0: 
                        return (nx, ny)
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return start_c

    def a_star(self, start_pos, goal_pos):
        """
        A* 尋路演算法
        
        實現細節：
        - 轉換像素座標為網格座標
        - 使用啟發式函數（曼哈頓距離）加速搜尋
        - 返回採樣後的路徑點（每 15 格採樣一次以減少中間點）
        
        參數：
            start_pos: (x, y) 起點像素座標
            goal_pos: (x, y) 終點像素座標
            
        返回值：
            從起點到終點的路徑點列表
        """
        start_c = (int(start_pos[0] // self.cell_size), int(start_pos[1] // self.cell_size))
        goal_c = (int(goal_pos[0] // self.cell_size), int(goal_pos[1] // self.cell_size))
        
        start_c = self.get_nearest_free_cell(start_c)
        goal_c = self.get_nearest_free_cell(goal_c)
        
        if not (0 <= start_c[0] < self.cols and 0 <= start_c[1] < self.rows): 
            return []
            
        frontier = []
        heapq.heappush(frontier, (0, start_c))
        came_from = {start_c: None}
        cost_so_far = {start_c: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_c: 
                break
            
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                next_c = (current[0] + dx, current[1] + dy)
                if 0 <= next_c[0] < self.cols and 0 <= next_c[1] < self.rows:
                    if self.grid[next_c] == 1: 
                        continue 
                    
                    # Diagonal movement through walls check
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
        
        if goal_c not in came_from: 
            return [goal_pos]
            
        path = []
        curr = goal_c
        while curr != start_c:
            path.append((curr[0] * self.cell_size + self.cell_size//2, curr[1] * self.cell_size + self.cell_size//2))
            curr = came_from[curr]
        path.reverse()
        
        # Sample path to reduce waypoints
        sampled_path = path[::15]
        if goal_pos not in sampled_path:
            sampled_path.append(goal_pos)
            
        return sampled_path


def generate_global_goals(center_x, corridor_w, pen_w, pen_h):
    """
    自動生成 8 個房間的目標位置
    
    口序：從下後序排紀（索引 0～7）
    - 索引 0, 1：下數第 4 間（後前各一個）
    - 索引 2, 3：下數第 3 間
    - 以此類推 ...
    - 索引 6, 7：上數第 1 間
    
    參數：
        center_x: 走道中心 x 座標
        corridor_w: 走道寬度
        pen_w, pen_h: 房間寬度、高度
        
    返回值：
        8 個目標位置的 [(x, y), ...] 列表
    """
    goals = []
    cl = center_x - corridor_w/2
    cr = center_x + corridor_w/2
    for i in range(3, -1, -1):
        y = i * (pen_h + 5) + 20 + pen_h/2
        goals.append((cl - pen_w/4, y))
        goals.append((cr + pen_w/4, y))
    return goals


def draw_static_obstacles(surface, screen_w, screen_h, center_x, corridor_w, 
                          pen_w, pen_h, door_w, bin_r):
    """
    繪製所有靜止障礙物
    
    繪製內容：
    - 走道最上数度頭：潦所佬 5 像素黑框
    - 4 個房間：左右各 2 個（共 8 間）
    - 每個房間：牆壁、靜頌門、緑色飼料桶（完羅元）
    
    參數：
        surface: pygame 表面物件
        screen_w, screen_h: 螢幕寬高
        center_x: 走道中心 x
        corridor_w: 走道寬度
        pen_w, pen_h: 房間寬高
        door_w: 門寶寬度
        bin_r: 飼料桶半徑
        
    返回值：
        房間訊息字典列表
        [
            {"bounds": (x_min, y_min, x_max, y_max), "bin_pos": (x, y)},
            ...
        ]
    """
    surface.fill((0, 0, 0, 0))
    pygame.draw.rect(surface, (0,0,0,255), (0, 0, screen_w, screen_h), 5)
    
    cl = center_x - (corridor_w / 2)
    cr = center_x + (corridor_w / 2)
    wall_color = (0, 0, 0, 255)
    wall_thick = 2
    
    pen_info = []

    for i in range(4):
        y = i * (pen_h + 5) + 20
        dy = y + pen_h / 2 - door_w / 2
        
        # Left pens
        pen_info.append({"bounds": (cl-pen_w, y, cl, y+pen_h), "bin_pos": (cl-pen_w/2, y+pen_h/2)})
        pygame.draw.line(surface, wall_color, (cl - pen_w, y), (cl, y), wall_thick)
        pygame.draw.line(surface, wall_color, (cl - pen_w, y), (cl - pen_w, y + pen_h), wall_thick)
        pygame.draw.line(surface, wall_color, (cl - pen_w, y + pen_h), (cl, y + pen_h), wall_thick)
        pygame.draw.line(surface, wall_color, (cl, y), (cl, dy), wall_thick)
        pygame.draw.line(surface, wall_color, (cl, dy + door_w), (cl, y + pen_h), wall_thick)
        pygame.draw.circle(surface, (0, 255, 0, 255), (cl - pen_w/2, y + pen_h/2), bin_r)

        # Right pens
        pen_info.append({"bounds": (cr, y, cr+pen_w, y+pen_h), "bin_pos": (cr+pen_w/2, y+pen_h/2)})
        pygame.draw.line(surface, wall_color, (cr, y), (cr + pen_w, y), wall_thick)
        pygame.draw.line(surface, wall_color, (cr + pen_w, y), (cr + pen_w, y + pen_h), wall_thick)
        pygame.draw.line(surface, wall_color, (cr, y + pen_h), (cr + pen_w, y + pen_h), wall_thick)
        pygame.draw.line(surface, wall_color, (cr, y), (cr, dy), wall_thick)
        pygame.draw.line(surface, wall_color, (cr, dy + door_w), (cr, y + pen_h), wall_thick)
        pygame.draw.circle(surface, (0, 255, 0, 255), (cr + pen_w/2, y + pen_h/2), bin_r)
    
    return pen_info
