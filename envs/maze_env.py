import numpy as np
import collections

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True):

        self.viewer = None
        self.enable_render = enable_render

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(200, 200), 
                                        enable_render=enable_render)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = 0
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(200, 200),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # execute action
        if isinstance(action, int):
            #print('This action was executed (int)', self.ACTION[action])
            authorized = self.maze_view.move_robot(self.ACTION[action])
        else:
            #print('This action was executed (non int)', action)
            if action == 0:
                action = 'N'
            elif action == 1:
                action = 'S'
            elif action == 2:
                action = 'E'
            elif action == 3:
                action = 'W'        
            authorized = self.maze_view.move_robot(action)
        obs = 0
        for elem in self.maze_view.obstacles: 
            if collections.Counter(elem) == collections.Counter(self.maze_view.robot): 
                obs = 1
        self.maze_view.moves+=1
        if not authorized: #if action is rejected
            reward = -10
            done = False
        else:
            if self.maze_view.moves<self.maze_view.budget:
                #print(np.sum(self.maze_view.states))
                if np.sum(self.maze_view.states) == 85: #reached goal
                    reward = 100*(np.sum(self.maze_view.states)/self.maze_view.moves)
                    done = True
                    print('     Won!')
                elif obs == 1: #reached an obstacle
                    reward = -30
                    done = False
                    #print('Hit obstacle', self.maze_view.robot)
                    self.maze_view.states[self.maze_view.robot[0], self.maze_view.robot[1]] = 1
                else:
                    if self.maze_view.states[self.maze_view.robot[1], self.maze_view.robot[0]] == 1: # already visited cell
                        reward = 0
                        done = False
                    else: # new  cell
                        reward = 10
                        done = False
                        #print('Step reward (game) 10')
                        self.maze_view.states[self.maze_view.robot[1], self.maze_view.robot[0]] = 1 # mark it as visited
            else: # out of moves
                reward = 100*(np.sum(self.maze_view.states)/self.maze_view.moves)
                done = True
                print('     Lost!')
        self.state = self.maze_view.robot


        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        self.maze_view.moves = 0
        self.maze_view.states = np.zeros(self.maze_size, dtype=int)
        self.maze_view.states[0][0] = 1
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), enable_render=enable_render)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", enable_render=enable_render)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), enable_render=enable_render)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render)

