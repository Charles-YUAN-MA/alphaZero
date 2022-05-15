from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque

from gomoku_gym.submission2_mcts_pure import MCTSPlayer as MCTS_Pure
from gomoku_gym.envs.game import Board, Game

from gomoku_alpha_zero.submission3_mcts_alphaZero import MCTSPlayer
from gomoku_alpha_zero.submission3_policy_value_net_pytorch import PolicyValueNet  


class TrainPipeline():
    def __init__(self, init_model=None):
        """ init function for the class"""

        # params of the board and the game
        self.board_width = 8 # board width
        self.board_height = 8 # board height
        self.n_in_row = 5 # win by n in line (vertically, horizontally, diagonally)
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5 # a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
        self.buffer_size = 10000 # buffer size for replaying experience
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size) # buffer
        self.play_batch_size = 1 # size of rollout for each episode
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02 # target of KL loss
        self.check_freq = 50 # frequency for check evaluation and save model
        self.game_batch_num = 1500 # number of training game loop
        self.best_win_ratio = 0.0 # best evaluated win ratio
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model: # load from existing file
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping

        Description:
            We can increase the training data by simply rotating or flipping the state. In such a way,
            we can get more data to contribute to increasing the performance of training neural network.

        input params:
            play_data: type:List,  [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play rollout data for training

        input param:
            n_games: number of rollout
        """
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net by training net

        Pipeline:
            1. sample data from the deque: self.data_buffer
            2. compute action probability for original policy network
            3. train neural network in a loop given sampled data
                    loop pipeline:
                        1. call self.policy_value_net.train_step(state_batch,
                                                                mcts_probs_batch,
                                                                winner_batch,
                                                                self.learn_rate*self.lr_multiplier)
                        2. compute action probability for new trained policy network
                        3. compute kl divergence between old and new action probability
                        4. if kl > self.kl_targ * 4, break the loop for Early Stopping
            4. adjust learning rate based on kl divergence
                    if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                        self.lr_multiplier /= 1.5 # decrease learning rate
                    elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                        self.lr_multiplier *= 1.5 # increase learning rate
            4. return final loss and entropy


        :return:
            loss:
            entropy:
        """
        loss, entropy = None, None
        # TODO: code here
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """ Policy Evaluation

        Description:
            Evaluate the trained policy by playing against the pure MCTS player
            Note: this is only for monitoring the progress of training

        Pipeline:
            1. create MCTSPlayer and MCTS_Pure Player
            2. Evaluation loop
                    Pipeline:
                        1. Rollout simulation for AlphaZero vs Pure MCTS
                                winner = self.game.start_play(current_mcts_player,
                                                  pure_mcts_player,
                                                  start_player=i % 2,  # start from either Player 1 or 2 evenly
                                                  is_shown=0)
                        2. Record result
            3. compute winning ratio: win_ratio
                    winning ratio =  (winning times + 0.5 * tie times) / total times
        return:
            win_ratio
        """
        # TODO: code here
        win_ratio =None
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio
        return win_ratio

    def run(self):
        """run the training pipeline

        Descriptions:
            train alpha zero in a loop.
            loop size: self.game_batch_num

        loop pipline:
            1. collect self-play data by rollouts
            2. policy update by sampled training data
            3. evaluated model performance (in a fixed frequency)
            4. save model (in a fixed frequency)
            5. evaluation result
        Plot

        """
        try:
           for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0            
            # TODO: code here
            #pass
        except KeyboardInterrupt:
            print('\n\rquit')

        # plot evaluation result
        # TODO: code here

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()