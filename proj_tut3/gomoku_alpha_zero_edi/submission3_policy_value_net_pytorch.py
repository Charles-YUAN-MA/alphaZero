import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        """ init function for the class

        Pipeline:
            1. create convolution layers for common layers:
                    1.common_layer1: nn.Conv2d(4, 32, kernel_size=3, padding=1)
                    2.common_layer2: nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    3.common_layer3: nn.Conv2d(64, 128, kernel_size=3, padding=1)
                (explaination for params of nn.Conv2d can be found in
                https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148)
            2. create layers for policy head
                    policy_layer1: nn.Conv2d(128, 4, kernel_size=1)
                    policy_layer2: nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
            3. create layers for value head
                    value_layer1: nn.Conv2d(128, 4, kernel_size=1)
                    value_layer2: nn.Linear(2*board_width*board_height, 64)
                    value_layer3: nn.Linear(64, 1)

        '"""
        super(Net, self).__init__()
        # TODO: program here
        """
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)
        """
        self.board_width = board_width
        self.board_height = board_height
        self.common_layer1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.common_layer2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.common_layer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.policy_layer1 = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_layer2 = nn.Linear(4*board_width*board_height, board_width*board_height)
        self.value_layer1= nn.Conv2d(128, 2, kernel_size=1)
        self.value_layer2= nn.Linear(2*board_width*board_height, 64)
        self.value_layer3= nn.Linear(64, 1)

    def forward(self, state_input):
        """ forward pass for policy net

        :param
            state_input: state input, 4D tensor in shape (batch_size, 4, board_height, board_width)

        :return: x_act, x_val
            x_act: action probability, 2D tensor in shape (batch_size, 4*board_height*board_width)
            x_val: value for state input, 2D tensor in shape (batch_size, 2*board_height*board_width)

        Pipeline:
            x = relu(common_layer1(x))
            x = relu(common_layer2(x))
            x = relu(common_layer3(x))

            x_act = relu(policy_layer1(x))
            x_act = reshape or flatten x_act from 4D tensor into 2D tensor in shape (batch_size, 4*self.board_width*self.board_height)
            x_act = log_softmax(policy_layer2(x_act))

            x_val = relu(value_layer1(x))
            x_val = reshape or flatten x_act from 4D tensor into 2D tensor in shape (batch_size, 2*self.board_width*self.board_height)
            x_val = relu(value_layer2(x_val))
            x_val = tanh(value_layer3(x_val))

            (Note: relu, tanh, log_softmax are activation functions)
        """
        x_act, x_val = None, None
        # TODO: program here
        """
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = torch.log_softmax(self.act_fc1(x_act),1,float)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        """
        x=state_input
        x = F.relu(self.common_layer1(x))
        x = F.relu(self.common_layer2(x))
        x = F.relu(self.common_layer3(x))
        x_act = F.relu(self.policy_layer1(x))
        x_act = x_act.view(-1, 4*self.board_height*self.board_width)
        x_act = torch.log_softmax(self.policy_layer2(x_act),1,float)
        x_val = F.relu(self.value_layer1(x))
        x_val = x_val.view(-1, 2*self.board_height*self.board_width)
        x_val = F.relu(self.value_layer2(x_val))
        x_val = torch.tanh(self.value_layer3(x_val))
        return x_act, x_val

        #return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        """ init function """
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """ get action probabilities and values given state batch

        input:
            state_batch: a batch of states,
                                shape: 4D array in (batch_num, 4, board_height, board_width)
        output: a batch of action probabilities and state values
            act_probs: a batch of probabilities,
                                shape: 4D array in (batch_num, 4, board_height, board_width)
            value: a vector of value
                                shape: 2D array in (batch_num, 1)
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """ get action probabilities and values given current board

        input:
            board
        return:
            a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """ perform a training step

        :param state_batch: a list of states,
                                element shape: 3D array in (4, board_height, board_width)
        :param mcts_probs: a list of mcts probabilites,
                                element shape: 1D array in (board_height*board_width,)
        :param winner_batch: a list of winning result: e.g. [-1,1,1,0,-1,.....]
        :param lr: learning rate
        :return: loss, entropy
            loss: training loss
            entropy: prediction KL divergence w.r.t. itself

        Pipeline:
            1. convert to torch tensor
                a. convert state_batch to 4D tensor in shape (batch_size, 4, board_height, board_width)
                b. convert mcts_probs to 2D tensor in shape (batch_size, board_height*board_width)
                c. convert winner_batch to 1D tensor in shape (batch_size,)
            2. self.optimizer.zero_grad()
            3. set learning rate by calling function: set_learning_rate(self.optimizer, lr)
            4. forward pass: log of act_probs, value = self.policy_value_net(state_batch)
            5. value loss =  (value - winner_batch)^2, mse_loss between value and winner_batch
            6. policy loss = -mcts_probs * log(act_probs), KL divergence loss between mcts_probs and act_probs
            7. Loss = value_loss + policy_loss
            8. optimize with loss by:
                    loss.backward()
                    self.optimizer.step()
            9. calculate act_probs KL divergence loss for act_probs w.r.t. itself : -act_probs * log(act_probs)
            10 return:  loss  and  act_probs KL divergence
        """
        # TODO: program here
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        #return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()
        """
        state_batch = Variable(torch.FloatTensor(state_batch))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs))
        winner_batch = Variable(torch.FloatTensor(winner_batch))
        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)
        logOfAct_probs, value = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)#5. value loss =  (value - winner_batch)^2, mse_loss between value and winner_batch
        policy_loss = torch.mean(torch.sum(-mcts_probs*logOfAct_probs, dim=1,dtype=float))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = torch.mean(torch.sum(-torch.exp(logOfAct_probs) * logOfAct_probs, dim=1,dtype=float))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)