import gym
import numpy as np
from abc import ABC
from gym import spaces
from typing import Tuple
from copy import deepcopy
from collections import namedtuple

Ship = namedtuple('Ship', ['min_x', 'max_x', 'min_y', 'max_y'])
Action = namedtuple('Action', ['x', 'y'])


# Extension: Add info for when the ship is sunk


class BattleshipEnv(gym.Env, ABC):
    def __init__(self, board_size: Tuple = None, ship_sizes: dict = None, episode_steps: int = 100):
        self.ship_sizes = ship_sizes or {5: 1, 4: 1, 3: 2, 2: 1}
        self.board_size = board_size or (10, 10)

        self.board = None
        self.board_generated = None
        self.observation = None

        self.done = None
        self.step_count = None
        self.episode_steps = episode_steps

        self.action_space = spaces.Discrete(self.board_size[0] * self.board_size[1])

        # MultiBinary is a binary space array
        self.observation_space = spaces.MultiBinary([2, self.board_size[0], self.board_size[1]])

        # dict to save all the ship objects
        self.ship_dict = {}

    def step(self, raw_action: int) -> Tuple[np.ndarray, int, bool, dict]:
        assert (raw_action < self.board_size[0]*self.board_size[1]),\
            "Invalid action (Superior than size_board[0]*size_board[1])"

        action = Action(x=raw_action // self.board_size[0], y=raw_action % self.board_size[1])
        self.step_count += 1
        if self.step_count >= self.episode_steps:
            self.done = True

        # it looks if there is a ship on the current cell
        # if there is a ship then the cell is 1 and 0 otherwise
        if self.board[action.x, action.y] != 0:

            # if the cell that we just hit is the last one from the respective ship
            # then add this info to the observation
            if self.board[self.board == self.board[action.x, action.y]].shape[0] == 1:
                ship = self.ship_dict[self.board[action.x, action.y]]
                self.observation[1, ship.min_x:ship.max_x, ship.min_y:ship.max_y] = 1

            self.board[action.x, action.y] = 0
            self.observation[0, action.x, action.y] = 1

            # if the whole board is already filled, no ships
            if not self.board.any():
                self.done = True
                return self.observation, 100, self.done, {}
            return self.observation, 1, self.done, {}

        # we end up here if we hit a cell that we had hit before already
        elif self.observation[0, action.x, action.y] == 1 or self.observation[1, action.x, action.y] == 1:
            return self.observation, -1, self.done, {}

        # we end up here if we hit a cell that has not been hit before and doesn't contain a ship
        else:
            self.observation[1, action.x, action.y] = 1
            return self.observation, 0, self.done, {}

    def reset(self):
        self.set_board()

        # maintain an original copy of the board generated in the start
        self.board_generated = deepcopy(self.board)
        self.observation = np.zeros((2, *self.board_size), dtype=np.float32)
        self.step_count = 0
        return self.observation

    def set_board(self):
        self.board = np.zeros(self.board_size, dtype=np.float32)
        k = 1
        for i, (ship_size, ship_count) in enumerate(self.ship_sizes.items()):
            for j in range(ship_count):
                self.place_ship(ship_size, k)
                k += 1

    def place_ship(self, ship_size, ship_index):
        can_place_ship = False
        while not can_place_ship:
            ship = self.get_ship(ship_size, self.board_size)
            can_place_ship = self.is_place_empty(ship)

        # set the ship cells to one
        self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y] = ship_index
        self.ship_dict.update({ship_index: ship})

    @staticmethod
    def get_ship(ship_size, board_size) -> Ship:
        if np.random.choice(('Horizontal', 'Vertical')) == 'Horizontal':
            # find the ship coordinates randomly
            min_x = np.random.randint(0, board_size[0] - 1 - ship_size)
            min_y = np.random.randint(0, board_size[1] - 1)
            return Ship(min_x=min_x, max_x=min_x + ship_size, min_y=min_y, max_y=min_y + 1)
        else:
            min_x = np.random.randint(0, board_size[0] - 1)
            min_y = np.random.randint(0, board_size[1] - 1 - ship_size)
            return Ship(min_x=min_x, max_x=min_x + 1, min_y=min_y, max_y=min_y + ship_size)

    def is_place_empty(self, ship):
        # make sure that there are no ships by simply summing the cell values
        return np.count_nonzero(self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y]) == 0

    def get_board(self):
        return self.board
