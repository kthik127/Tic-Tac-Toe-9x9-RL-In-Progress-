# for the two neural networks, qlearning starts here

"""
two networks: q_target, which is frozen snapshot of q_online,
q_target is used in the  equation to train q_online and calculate bellman target and Q(s | a)
don't want model to be using a neural network function with constantly changing weights
so q_target is snapshot of neural network
using bellman qlearning equation, putting negative weight if reward is negative, calculating Q value based on state and action and reward
greedy optimization, and taking action that produces max Q-value (based on reward weight)
uses e-greedy exploitation and action masking (blocking illegal moves)
agent.py will be used with memory buffer in train.py to train the neural network (train.py runs actual simulations) 
"""

import torch
import torch.nn as nn
from network import DQN
import random

class DQNAgent:

    def __init__(self,gamma=0.99, lr=0.0001, target_sync_interval=1000):
        """
        gamma: discount factor
        lr: learning rate
        target_sync_interval: frequency of which we copy q_online weights into q_target (represented as matrices)
        """
        self.gamma = gamma
        self.target_sync_interval = target_sync_interval
        self.q_online = DQN()
        self.q_target = DQN() # a duplicate DQN
        self.q_target.load_state_dict(self.q_online.state_dict()) # copying ALL the weights of q_online as single dictionary into q_target, not used for training
        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr) # updates weights during training
        self.loss_fn = nn.MSELoss() # mean squared error
        self.step_counter = 0

    def select_action(self, state, legal_moves, epsilon): # for choosing actions
        """pass"""
        # explore branch, selects a random choice from legal moves with probability epsilon (e-greedy algorithm)
        if random.random() < epsilon:
            return random.choice(legal_moves)
        # action masking (for restricted select action to LEGAL moves)

        q_values = self.q_online(state.unsqueeze(0))  # automatically calls the forward() method internally, same as self.q_online.forward
        mask = torch.full((81,), float('-inf'))
        mask[legal_moves] = 0.0
        masked_q = q_values + mask
        return masked_q.argmax().item()

    def train_step(self, buffer, batch_size): # the actual training for the q-learning function
        """pass"""

        if len(buffer) < batch_size: # not enough data
            return

        batch = buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch) # unpacking into different lists for EACH feature
        # these are already tensors
        states_tensor = torch.stack(states)
        next_states_tensor = torch.stack(next_states)

        # these are just simple tuples, convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_online(states_tensor) # getting q values from the neural network function
        q_val = q_values.gather(1,actions_tensor.unsqueeze(1)).squeeze(1) # getting the specific q-val from the ACTION transition taken (in actions_tensor)

        # compute the bellman target

        with torch.no_grad(): # to disable extra gradient calculations while calculating q-value
            q_next = self.q_target(next_states_tensor) # getting q values from the snapshot neural network with the next-state tensor  for reference
            q_val_next = q_next.max(1)[0] # finding the max q-value
            bellman_target =  rewards_tensor + self.gamma * q_val_next * (1 - dones_tensor)  # calculating bellman target, only continue if done is not true yet, which makes 1-done = 0

        loss = self.loss_fn(q_val, bellman_target) # comparing predicted q_val with the correct q_val (bellman_target)

        self.optimizer.zero_grad()
        loss.backward() # accumulates gradients
        self.optimizer.step()

        self.step_counter += 1  # increment step counter
        if self.step_counter % self.target_sync_interval == 0: # time to sync
            self.q_target.load_state_dict(self.q_online.state_dict()) # copying weights again








