from gym import Env, spaces
from gym.utils import seeding
from importlib_metadata import pass_none
from gomoku_gym.envs.game import Board, Game
import numpy as np
# from gomoku_gym.mcts_pure import MCTSPlayer


class GameEnv(Env):
    def __init__(self, width, height, n_in_row) -> None:
        self.board = Board(width=width, height=height, n_in_row=n_in_row)
        self.height = height
        self.width = height
        self.n_in_row = n_in_row
        self.p1, self.p2 = self.board.players

        # Observation space on board
        shape = (self.height, self.width) # board_size * board_size
        self.observation_space = spaces.Box(np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=np.float32))
        
        # One action for each board position
        self.action_space = spaces.Discrete(int(self.height)*int(self.width))
    
    def step(self, action):
        done = False
        info = {}
        reward = 0


        current_player = self.board.get_current_player()
        self.board.do_move(action)
        end, winner = self.board.game_end()
        if end:
            info = {'winner': winner}
            done = True
            if winner ==-1: #tie
                reward = 0
            elif winner == current_player:
                reward = 1
            else:
                reward = -1

        obs = self._get_obs()
        return obs, reward, done, info
    
    def _get_obs(self):
        return self.board.current_state()
    def reset(self, start_player=0):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        obs = self._get_obs()
        return obs


    def render(self):
        """Draw the board and show game info"""
        width = self.board.width
        height = self.board.height

        print("Player", self.p1, "with X".rjust(3))
        print("Player", self.p2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = self.board.states.get(loc, -1)
                if p == self.p1:
                    print('X'.center(8), end='')
                elif p == self.p2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')


# if __name__ == '__main__':
#     n_in_row = 5
#     width, height = 8, 8
#     is_humanMoveFirst = True
#     test_player = MCTSPlayer(c_puct=5, n_playout=1000)
#     oponent_player = MCTSPlayer(c_puct=5, n_playout=1000)
#     env = GameEnv(width, height, n_in_row)
#     p1, p2 = env.board.players
#     test_player.set_player_ind(p1)
#     oponent_player.set_player_ind(p2)
#     obs = env.reset()
#     done = False
#     while not done:
#         if env.board.get_current_player() == p1:
#             action = test_player.get_action(env)
#         else:
#             action = oponent_player.get_action(env)
#         obs, reward, done, info = env.step(action)
#         env.render()
#         print(done)