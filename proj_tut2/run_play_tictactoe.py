from gomoku_gym.envs.game_env import GameEnv
from gomoku_gym.envs.wrappers import GUI_Wrapper
from gomoku_gym.submission2_mcts_pure import MCTSPlayer


# parameters (you can test other environment parameters)
n_in_row = 5
width, height = 8, 8
is_humanMoveFirst = True


# run
oponent_player = MCTSPlayer(c_puct=5, n_playout=1000)
env = GameEnv(width, height, n_in_row)
obs = env.reset()
env = GUI_Wrapper(env)
env.run(p2=oponent_player)