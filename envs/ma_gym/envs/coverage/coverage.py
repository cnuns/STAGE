# Using local gym
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

try:
    from ..utils.action_space import MultiAgentActionSpace
    from ..utils.observation_space import MultiAgentObservationSpace
    from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text, draw_sensing_outline
except:
    sys.path.append(current_file_path + '/../../../../')
    sys.path.append(current_file_path + '/../')
    from envs.utils.action_space import MultiAgentActionSpace
    from envs.utils.observation_space import MultiAgentObservationSpace
    from envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text, draw_sensing_outline

from custom_implement.env_communication import init_communication, print_env_params, update_communication_state
from custom_implement.connection_check import is_connected

import time
import random
import re
import pickle
import torch


logger = logging.getLogger(__name__)

class Coverage(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, full_observable: bool = False, step_cost: float = 0, n_agents: int = 3, max_steps: int = 400, clock: bool = False, **kwargs):
        params = p = kwargs['params']
        
        m = params.get("grid_size")
        self.original_map = 10
        self._grid_shape = (m, m)
        self._grid_available_x = (0, self._grid_shape[0]-1)
        self._grid_available_y = (0, self._grid_shape[1]-1)
        self._max_steps = self.Tmax = params.get("max_env_steps")
        self._step_count = None

        self.n_color = self.n_groups = params.get("n_groups")
        self.n_agents = params.get("n_agents")
        self.n_nodes = int(self.n_agents / self.n_groups) 
        assert self.n_agents == self.n_groups*self.n_nodes

        self._add_clock = clock if params.get("add_clock") is None else params.get("add_clock")

        self.MapRatio = MapRatio = m / self.original_map
        
        self.agg = sum
        self.load = params.get("load")
        self.n_obstacles = self.load * int(self.MapRatio ** 2) # map 사이즈에 따른 obstacle 개수

        self.rendering = params.get("rendering", False)
        self.sorted_poses = self.get_sorted_poses()

        self.obst_box = np.ones((2, 2))
        
        assert MapRatio == int(MapRatio)
        r = int(MapRatio)

        self.first_place_obstacles = True
        self._total_episode_reward = None
        
        #! different by scenario, but must fill in
        self.capture_reward = abs(params.get("capture_reward"))
        self._step_cost = -abs(params.get("step_cost"))
        self.move_penalty = -abs(params.get("rm"))
        self.penalty = -abs(params.get("penalty"))
        self.revisit_penalty = -abs(params.get("revisit_penalty"))
        self.lazy_penalty = -abs(params.get("lazy_penalty"))
        self.final_reward = 0 # self.capture_reward*(m**2) * abs(params.get("final_reward"))

        self.bound_return = None
        self.maps = params.get("grid_size")
        self.Rsen = params.get("Rsen")
        self._agent_view_mask = (2*self.Rsen+1, 2*self.Rsen+1)
        init_communication(self, params)

        self.n_min = params.get("n_min") #number of nearest agents
        if self.mode in ['train', 'restore']:
            self.R_com = params.get("trRcom")
        else:
            self.R_com = params.get("teRcom")
        x = torch.FloatTensor([[0,0], [self.R_com,self.R_com]])
        distances = torch.cdist(x, x)
        self.com_range = self.Rcom_th.item() if self.Rcom_th.item() else distances[0][-1].item()

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])  # l,r,t,d,noop
        
        self.first = True
        self._full_obs = self.__create_grid()
        self.viewer = None

        self.full_observable = full_observable
        self.define_obs_space()
        param_print = params.get("env_param_print", 1)
        print_env_params(self, option=param_print)

    def define_obs_space(self):
        # vision:(3, Rsen x Rsen) + x, y
        depth = 3
        vision = np.prod(self._agent_view_mask)

        self._obs_high = np.array([1.]*depth*vision + [1., 1.] * self.n_min)
        self._obs_low = np.array([-1.]*depth*vision + [0., 0.] * self.n_min)

        if self._add_clock:
            self._obs_low = np.concatenate((self._obs_low, np.array([0.])))
            self._obs_high = np.concatenate((self._obs_high, np.array([1.])))

        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
            
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if self.__wall_exists((row, col)):
                    fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=WALL_COLOR, margin=0.1)

        # Draw Empty Cell
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if self.__visited((row, col)):
                    color = self._cover_channel[row][col]
                    fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=COLORS[color], margin=0.1)

    def __create_grid(self):
        self._base_grid = np.zeros(self._grid_shape)
        _grid = np.full((self._grid_shape[0], self._grid_shape[1]), '0', dtype=str)

        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        self._obst_map = np.zeros(self._full_obs.shape)
        self._visited = np.zeros(self._full_obs.shape)
        self._cover_channel = np.full(self._full_obs.shape, PRE_IDS['empty'], dtype=str)

        self._full_obs = self._full_obs.tolist()

        for agent_i in range(self.n_agents):
            group_i = self.agent_team[agent_i]

            pos = self.sorted_poses[agent_i]
            self.agent_pos[agent_i] = pos
            self._visited[pos[0]][pos[1]] = 1

            color = PRE_IDS[f'group{group_i}']
            self._cover_channel[pos[0]][pos[1]] = color

            self.__update_agent_view(agent_i)
            self.__update_obstacle_view(self._obst_map, np.array([[1]]), pos, margin=1) #agent 근처 1칸 이내에는 obstacle 못 둠.
        
        self.__create_obstacle()
        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(0, self.n_agents):
            pos = self.agent_pos[agent_i]
            local_agent_view = self.get_local_view(center_pos=pos)

            _agent_i_obs = []
            _agent_i_obs += local_agent_view.flatten().tolist() # local vision: 3x3 for each [space, agents, goals] --> (3, 5x5)
            _agent_i_obs += self.get_nearest_agents_pos(agent_i, self.com_range, self.n_min) # adding nearest agents pos
            if self._add_clock:
                _agent_i_obs += [self._step_count / self._max_steps]  # time

            _obs.append(_agent_i_obs)

        return _obs

    def get_bound_return(self):
        if self.agg == mean:
            r = self.capture_reward*(self.n_empty_cells)/self.n_agents - abs(self._step_cost)*(self.n_empty_cells)/self.n_agents + self.final_reward
        elif self.agg == sum:
            r = self.capture_reward*(self.n_empty_cells) + self.final_reward
        return r
    
    def reset(self, epoch=-1):
        agent_nodes_split = num_of_group_split(self.n_agents, self.n_groups)
        self.agent_team, self.agent_groups = divide_n_group(total_nodes=self.n_agents, n_group=self.n_groups, n_nodes=agent_nodes_split)

        self.__init_full_obs()

        self.total_capture_cnt = 0
        self.n_empty_cells = np.count_nonzero(self._obst_around_grid)

        self.ave_trput = self.n_empty_cells
        
        self.bound_return = self.get_bound_return()

        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._total_episode_reward = 0 #= [0 for _ in range(self.n_agents)]

        update_communication_state(self)

        if self.rendering:
            self.render()
            time.sleep(1)

        return self.get_agent_obs()

    def __is_valid(self, pos, margin=0):
        return (0+margin <= pos[0] < self._grid_shape[0]-margin) and (0+margin <= pos[1] < self._grid_shape[1]-margin)

    def __wall_exists(self, pos):
        row, col = pos
        return self._base_grid[row][col] == 1
    
    def __visited(self, pos):
        row, col = pos
        return self._visited[row][col] == 1

    def _is_cell_vacant(self, pos):
        return self.__is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])
    
    def _is_cell_vacant_obst(self, pos, obst):
        if not self.__is_valid((pos[0]+obst.shape[0]-1, pos[1]+obst.shape[1]-1)): #우하단 좌표가 valid하지 않으면 놓을 수 없음
            return False
        poses = [self._obst_map[pos[0]+i][pos[1]+j] for i in range(obst.shape[0]) for j in range(obst.shape[1]) if obst[i][j] == 1] #장애물 내 모든 좌표 상태 가져오기
        return self.__is_valid(pos) and (not any(poses)) #하나라도 1이면 놓을 수 없음

    def _is_possible_to_go(self, pos):
        return self.__is_valid(pos) and (self._full_obs[pos[0]][pos[1]] not in [PRE_IDS['wall'], 'R1', 'G1', 'B1'] )
    
    def get_empty_cell_positions(self, center_pos, BoxSize):
        cnt = 0
        positions = []
        dy = dx = int(BoxSize/2)
        for row in range(center_pos[0] - dy, center_pos[0] + dy + 1):
            for col in range(center_pos[1] - dx, center_pos[1] + dx + 1):
                px, py = row - (center_pos[0] - dy), col - (center_pos[1] - dx)
                if self._is_cell_vacant([row, col]):
                    cnt += 1
                    positions.append([row, col])
        
        return positions

    def __update_agent_view(self, agent_i):
        x, y = self.agent_pos[agent_i]
        self._full_obs[x][y] = PRE_IDS['agent'] + str(agent_i + 1)#agent_i + 1

    def __update_object_view(self, obj_symbol, pos, i=None):
        x, y = pos
        postfix = str(i + 1) if i !=None else ''
        self._full_obs[x][y] = PRE_IDS[obj_symbol] + postfix#agent_i + 1

    def __update_object_view_background(self, bg, obj_symbol, pos, i=None):
        x, y = pos
        postfix = str(i + 1) if i !=None else ''

        if type(obj_symbol) == str:
            bg[x][y] = PRE_IDS[obj_symbol] + postfix#agent_i + 1
        else:
            bg[x][y] = obj_symbol

    def __update_obstacle_view(self, obst_map, obst, pos, margin):
        for i in range(obst.shape[0]+2*margin):
            for j in range(obst.shape[1]+2*margin):
                if not (margin == 0 and obst[i][j] == 0): #margin 없으면서 장애물이 아닌 경우에는 표시 안 함
                    obst_map[pos[0]-margin+i][pos[1]-margin+j] = 1 #방어막(margin)까지 표시

    def __update_obstacle_around_grid(self, base_grid, obst, pos):
        for i in range(obst.shape[0]):
            for j in range(obst.shape[1]):
                if obst[i][j] == 1: #obstacle이면 주위에 cell 표시(obstacle 없으면 1이고 있으면 0)
                    self._obst_around_grid[pos[0]+i-1][pos[1]+j] = 1 if not base_grid[pos[0]+i-1][pos[1]+j] else 0 #상
                    self._obst_around_grid[pos[0]+i+1][pos[1]+j] = 1 if not base_grid[pos[0]+i+1][pos[1]+j] else 0 #하
                    self._obst_around_grid[pos[0]+i][pos[1]+j-1] = 1 if not base_grid[pos[0]+i][pos[1]+j-1] else 0 #좌
                    self._obst_around_grid[pos[0]+i][pos[1]+j+1] = 1 if not base_grid[pos[0]+i][pos[1]+j+1] else 0 #우

    def __is_agent_done(self, agent_i):
        return self.agent_pos[agent_i] == self.final_agent_pos[agent_i]

    def get_reward(self, capture_cnt, moving_cnt, penalty_cnt, lazy_cnt, revisit_cnt, final_reward):
        rewards = self._step_cost\
            + self.capture_reward*self.agg(capture_cnt)\
            + self.move_penalty*self.agg(moving_cnt)\
            + self.penalty*self.agg(penalty_cnt)\
            + self.lazy_penalty*self.agg(lazy_cnt)\
            + self.revisit_penalty*self.agg(revisit_cnt)\
            + final_reward

        reward_details = dict(
                            reward=rewards,
                            capture_cnt=self.agg(capture_cnt),
                            step_cnt=1,
                            move_cnt=self.agg(moving_cnt),
                            penalty_cnt=self.agg(penalty_cnt),
                            variable=self.agg(revisit_cnt),
                            vars2=self.agg(lazy_cnt))

        return rewards, reward_details
        
    def step(self, agents_action):
        self._step_count += 1
        step_count = 1

        capture_cnt = [0]*self.n_agents
        moving_cnt = [1 if action != ACTION_INDICES['NOOP'] else 0 for action in agents_action]
        penalty_cnt = [0]*self.n_agents
        lazy_cnt = [0]*self.n_agents
        revisit_cnt = [0]*self.n_agents
        final_reward = 0

        for agent_i, action in enumerate(agents_action):
            group_i = self.agent_team[agent_i]
            group_agent_i = 0

            curr_pos = copy.copy(self.agent_pos[agent_i])
            next_pos = None
            if action == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif action == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif action == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif action == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif action == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')

            if next_pos is None:
                # -1 for an action that results in no motion (lazy penalty).
                lazy_cnt[agent_i] += 1

            else: #next_pos is not None:
                if self._is_cell_vacant(next_pos):
                    self.agent_pos[agent_i] = next_pos

                    if (self._visited[next_pos[0]][next_pos[1]] == 0) and (self._obst_around_grid[next_pos[0]][next_pos[1]]): #visit하지 않으면서 obstacle 주위인 셀(unvisit & incentive)
                        self._visited[next_pos[0]][next_pos[1]] = 1

                        color = PRE_IDS[f'group{self.agent_team[agent_i]}']
                        self._cover_channel[next_pos[0]][next_pos[1]] = color # mark as first visited agent color
                        capture_cnt[agent_i] += 1 # +2 for moving to a previously unexplored cell (white).
                        
                        if self.rendering:
                            fill_cell(self._base_img, (next_pos[0],next_pos[1]), cell_size=CELL_SIZE, fill=COLORS[color], margin=0.1)

                    elif (self._visited[next_pos[0]][next_pos[1]] == 1) and (self._obst_around_grid[next_pos[0]][next_pos[1]]): #visit하면서 obstacle 주위인 셀(visit & incentive) -> 재방문
                        revisit_cnt[agent_i] += 1 # -0.5 for moving to an explored cell.

                    else: #visit하면서 obstacle 주위가 아닌 셀(visit & non incentive)
                        if self._visited[next_pos[0]][next_pos[1]] == 0: # visited 맵 업데이트
                            self._visited[next_pos[0]][next_pos[1]] = 1

                    self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty'] # mark as first visited agent color
                    self.__update_agent_view(agent_i)
                else:
                    # -1 for an illegal action (attempt to move outside the boundary or collide against other robots and obstacles).
                    penalty_cnt[agent_i] += 1

        #! map all explored
        self.total_capture_cnt +=  sum(capture_cnt)
        if self.total_capture_cnt == self.n_empty_cells:
            final_reward = self.final_reward
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        #! time expired
        if self._step_count >= self._max_steps:
            if all(self._agent_dones): self.success = 1
            else: self.success = 0
            
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        update_communication_state(self)

        rewards,reward_details = self.get_reward(capture_cnt, moving_cnt, penalty_cnt, lazy_cnt, revisit_cnt, final_reward)

        self._total_episode_reward += rewards

        if self.rendering:
            self.render()

        return self.get_agent_obs(), (rewards,reward_details), self._agent_dones, {}

    def render(self, mode='human'):
        if self.rendering == False:
            # Draw Empty Cell
            for row in range(self._grid_shape[0]):
                for col in range(self._grid_shape[1]):
                    if self.__visited((row, col)):
                        color = self._cover_channel[row][col]
                        fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=COLORS[color], margin=0.1)
            self.rendering = True

        img = copy.copy(self._base_img)

        # Draw agents
        for agent_i in range(self.n_agents):
            group_i = self.agent_team[agent_i]
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=GROUP_COLORS[group_i], radius=0.3)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
            
            #! draw sensing range
            pos = self.agent_pos[agent_i]
            row, col = pos[0], pos[1]
            draw_sensing_outline(img, (row, col), Rsen=self.Rsen, cell_size=CELL_SIZE, fill=AGENT_SENSING_COLOR, width=2)
            
        img = np.asarray(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_local_view(self, center_pos):
        """
            When expressing Rsen as the grid vertical length,
            That is, when the agent is always positioned at the center of the view,
            A function that marks and returns prey locations within Rsen cells around the agent.
        """
        depth = 3
        local_agent_view = np.zeros((depth, *self._agent_view_mask)) # 3 x 5 x 5  = RGB x H x W 
        dy = dx = self.Rsen
        own_pos = int(self._agent_view_mask[0] / 2), int(self._agent_view_mask[1]/2)
        for row in range(center_pos[0] - dy, center_pos[0] + dy + 1):
            for col in range(center_pos[1] - dx, center_pos[1] + dx + 1):
                px, py = row - (center_pos[0] - dy), col - (center_pos[1] - dx)
                # if (px, py) == own_pos:
                #     continue
    
                if not self.__is_valid([row, col]):
                    # means wall
                    local_agent_view[ID_WALL][px, py] = 1 # get relative vision for the wall loc.
                    local_agent_view[ID_VISITED][px, py] = 1 # cannot visit wall coordinates.
                else:
                    # elements
                    obj = self._full_obs[row][col]
                    
                    if obj == PRE_IDS['wall']:
                        local_agent_view[ID_WALL][px, py] = 1 # get relative vision for the wall loc.

                    if PRE_IDS['agent'] in obj:
                        local_agent_view[ID_AGENT][px, py] = 1 # get relative vision for the friend agent loc.

                    if not ((self._visited[row][col] == 0) and (self._obst_around_grid[row][col] == 1)): # (visit하지 않은 incentive cell)이 아니면 모두 1
                        local_agent_view[ID_VISITED][px, py] = 1 # get relative vision for the explored cell
                    
        return local_agent_view

    def get_nearest_agents_pos(self, my_id, com_range, n_min):
        """
        com_range 내의 에이전트 중 가장 가까운 n_min개를 찾아 [r, theta, r, theta...] 형태로 반환.
        에이전트가 부족할 경우 남은 자리는 0으로 채움.
        """
        # 1. 데이터를 넘파이 배열로 변환
        # self.agent_pos가 {id: [x, y]} 형태
        coords = np.array(list(self.agent_pos.values()), dtype=float)
        
        # 2. 상대 좌표 계산 (나 자신을 0, 0으로)
        self_coord = coords[my_id]
        relative_coords = coords - self_coord
        
        # 3. 극좌표 변환
        r = np.hypot(relative_coords[:, 0], relative_coords[:, 1])
        theta = np.degrees(np.arctan2(-relative_coords[:, 0], relative_coords[:, 1])) % 360
        
        # Normalization
        r /= com_range
        theta /= 360
        
        # 4. 필터링: com_range 이내이면서 자기 자신이 아닌 것 (r > 0)
        mask = (r <= 1) & (r > 0)
        filtered_r = r[mask]
        filtered_theta = theta[mask]
        
        # 5. 거리 순 정렬 및 상위 n_min개 인덱스 추출
        sorted_indices = np.argsort(filtered_r)
        closest_indices = sorted_indices[:n_min]
        
        # 6. 결과 리스트 생성
        nearest_agents = []
        for idx in closest_indices:
            nearest_agents.append(filtered_r[idx])
            nearest_agents.append(filtered_theta[idx])
            
        # 7. 패딩(Padding): 부족한 만큼 0으로 채우기
        needed_padding = n_min*2 - len(nearest_agents)
        if needed_padding > 0:
            nearest_agents.extend([0.0] * needed_padding)
                
        return nearest_agents

    def __create_obstacle(self):
        if self.n_obstacles != 0:
            self._obst_map[[0, -1], :] = 1 #9x9 맵에만 장애물 놓음
            self._obst_map[:, [0, -1]] = 1 #9x9 맵에만 장애물 놓음

            original_obst_map = copy.deepcopy(self._obst_map)
            original_base_grid = copy.deepcopy(self._base_grid)
            original_full_obs = copy.deepcopy(self._full_obs)
            self._obst_around_grid = np.zeros(self._grid_shape)
            while True:
                for obst_i in range(self.n_obstacles): #맵 사이즈에 따라 장애물 개수 조절
                    while True:
                        pos = [random.randint(*self._grid_available_x), random.randint(*self._grid_available_y)]
                        if self._is_cell_vacant_obst(pos, self.obst_box):
                            break
                    self.place_obstacle(pos, self.obst_box)
                    self.__update_obstacle_view(self._obst_map, self.obst_box, pos, margin=0) #margin: 0 --> no virtual block
                    self.__update_obstacle_around_grid(self._base_grid, self.obst_box, pos)
                
                if is_connected(self._base_grid):
                    break
                else:
                    self._obst_map = copy.deepcopy(original_obst_map)
                    self._base_grid = copy.deepcopy(original_base_grid)
                    self._full_obs = copy.deepcopy(original_full_obs)
                    self._obst_around_grid = np.zeros(self._grid_shape)

        return
    
    def place_obstacle(self, pos, shape):
        for i, row in enumerate(shape):
            for j, col in enumerate(row):
                dot = (pos[0]+i, pos[1]+j)
                if self.__is_valid(dot):
                    if shape[i, j] == 1:
                        self.__update_object_view('wall', dot)
                        self.__update_object_view_background(self._base_grid, 1, dot)
    
    def get_sorted_poses(self):
        center = torch.tensor(self._grid_shape, dtype=torch.float32) / 2 - 0.5 if self._grid_shape[0] % 2 == 0 else torch.tensor(self._grid_shape, dtype=torch.float32) // 2 #중앙 좌표 (10 --> 4.5, 15 --> 7, 20 --> 9.5)
        poses = torch.tensor([[i, j] for i in range(self._grid_available_x[0], self._grid_available_x[1]+1) for j in range(self._grid_available_y[0], self._grid_available_y[1]+1)]) #맵의 모든 좌표
        dists = torch.norm(poses - center, dim=1) #중앙 좌표와의 거리

        return poses[torch.argsort(dists)].tolist() #거리 작은 순으로 정렬


def num_of_group_split(num, n_groups):
    mod = num % n_groups
    quotient = num // n_groups

    assert quotient * n_groups + mod == num
    groups_num = [quotient]*n_groups
    
    for i in range(mod):
        groups_num[i] = groups_num[i] + 1

    return groups_num

def divide_n_group(total_nodes, n_group, n_nodes):
    # assert total_nodes == n_group * n_nodes
    # Generate an array of all node indices.
    all_nodes = np.arange(total_nodes)
    node_team = np.arange(total_nodes)

    # Initialize an array to store nodes belonging to each group.
    groups = np.empty(n_group, dtype=object)

    # Assign nodes to groups.
    for group_index in range(n_group):
        idx = np.random.choice(all_nodes, size=n_nodes[group_index], replace=False)
        groups[group_index] = idx
        node_team[idx] = group_index
        all_nodes = np.setdiff1d(all_nodes, groups[group_index])

    return node_team, groups

rAgent = re.compile(r'(R|G|B)[0-9]+')
GROUP_COLORS = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'orange'
}

COLORS = {
    'R': (241, 95, 95),#'red',
    'G': (134, 229, 127), #'green',
    'B': (92, 209, 229),# 'blue',
    'O': 'orange'
}

AGENT_COLORS = {
    0: 'red',
    1: 'green', 
    2: 'blue',
    3: 'orange'
}

CELL_SIZE = 30

WALL_COLOR = 'black'

AGENT_SENSING_COLOR = (47,157,39)# ImageColor.getcolor('green', mode='RGB')

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

ACTION_INDICES = {
    'DOWN': 0,
    'LEFT': 1,
    'UP': 2,
    'RIGHT': 3,
    'NOOP': 4,
}

PRE_IDS = {
    'agent': 'A',
    'group0': 'R',
    'group1': 'G',
    'group2': 'B',
    'goal': 'G',
    'wall': 'W',
    'empty': '0'
}

# COLORS = {
#     'B': np.array([0, 0, 0]) / 255,
#     'W': np.array([255, 255, 255]) / 255,
#     'R': np.array([255, 0, 0]) / 255,
#     'G': np.array([0, 255, 0]) / 255,
#     'B': np.array([0, 0, 255]) / 255,
#     'Y': np.array([255, 255, 0]) / 255,
# }

ID_WALL = 0
ID_AGENT = 1
ID_VISITED = 2
ID_GOAL = 2

def mean(x):
    return sum(x)/len(x)

def fix_randomness(seed):
    import torch

    #* randomness set
    deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    params = {
        'capture_reward': 2,
        'step_cost': -0.1,
        'rm': 0,
        'penalty': 1,
        'revisit_penalty': 0,
        'lazy_penalty': -1,
        
        'Rsen': 1,
        'env_param_print': 1,
        'n_groups': 3,
        'n_nodes': 1,
        'n_agents': 3,
        'n_targets': 2,
        'final_reward':100,

        'grid_size': (30),
        'obstComplex': 'Easy',
        'load':3,
        'n_obstacles': 2,
        
        'n_gcn_layers': 2,
        'curriculum_learning': 0,
        'channelType': 'FC',
        'loss_apply': 1,
        'Pgb': None,
        'Pbg': None,
        'calc_diameter': None,
        'mode': 'train',
        'trRcom': 0,
        'teRcom': 0,
        'trplPmin': None,
        'trplPmax': None,
        'trpltype': 'iid',
        'trpl':0,
        'trpfNmin': None,
        'trpfNmax': None,
        'trpftype': 'iid',
        'trpf':0,
        'fault_mode': None,
    }

    fix_randomness(1)
    
    def Net(obs):
        actions = selected_elements = np.random.choice(acts, size=params['n_agents'])
        return actions
    
    env = Coverage(params=params)
    obs = env.reset()
    env.render()
    time.sleep(1.0)
    acts = list(ACTION_MEANING.keys())
    for i_eps in range(10000):
        for i in range(400):
            # time.sleep(0.1)
            actions = Net(obs)
            obs, (rewards,reward_details), dones, _ = env.step(actions)

            if all(dones):
                env.render()
                time.sleep(1.0)

                obs = env.reset()
                env.render()
                time.sleep(1.0)
            else:
                env.render()
                pass
            
        