from gomoku_gym.envs.game_env import GameEnv
from gomoku_gym.envs.wrappers import GUI_Wrapper
# from gomoku_gym.submission2_mcts_pure import MCTSPlayer

from gomoku_alpha_zero.submission3_mcts_alphaZero import MCTSPlayer
from gomoku_alpha_zero.submission3_policy_value_net_pytorch import PolicyValueNet
from os.path import join

# parameters (you can test other environment parameters)
n_in_row = 4
width, height = 6, 6



# run
model = PolicyValueNet(board_width=width, 
                        board_height=height,
                        model_file=join(".", "best_policy.model"))
oponent_player = MCTSPlayer(policy_value_function=model.policy_value_fn,  c_puct=5, n_playout=1000)
env = GameEnv(width, height, n_in_row)
obs = env.reset()
env = GUI_Wrapper(env)
env.run(p2=oponent_player)