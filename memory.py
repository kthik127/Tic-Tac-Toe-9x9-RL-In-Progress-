# replay buffer memory (for memory of board states from previous games for non-randomized learning) DURING TRAINING ONLY

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        """single attribute: a deque storing moveplays"""
        # initialize deque (drops the transitions at the front once capacity is reached when new element is added in O(1) time
        self.transitions = deque(maxlen = capacity)
        # the transitions list is STORING move plays from all games for memory
        # dropping the least recent one, and random sample gets a collection of random states

    def push(self, state, action, reward, next_state, done):
        """
        adds a moveplay to the transitions
        moveplay ==> 5D tensor (board_state, action, reward, next_board_state, done)
        board state also represented as (4,9,9) tensor 
        """
        self.transitions.append((state,action,reward,next_state,done)) # adding a moveplay to our transitions list

    def sample(self, batch_size):
        """returns a random collection of moveplays"""
        return random.sample(self.transitions, batch_size) # returning a random collection of moveplays

    def __len__(self):
        "returns the length of the number of moveplays stored"
        return len(self.transitions)
