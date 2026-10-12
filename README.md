# Tic-Tac-Toe-9x9-RL (In Progress) 

An implementation of a Deep Q-Learning (DQN) agent applied to Ultimate 9x9 Tic Tac Toe, using PyTorch. 

### Game Background ### 

Ultimate 9x9 Tic Tac Toe is an extension of the original 3x3 Tic Tac Toe, with a 3x3 board of macro squares and each macro square being a 3x3 local square board. Each time a player chooses a local square, the next player is forced to choose a local square only in the corresponding macro square index, unless that square is won or drawn (indicating a free choice decision for the next player to choose any macro square on the board). A macro square is won like any board in 3x3 Tic Tac Toe, and the player with the most macro squares won wins the game. 

### Status ### 

Completed game environment rules, coordinate mappings for game positions, convolutional neural network (CNN) framework, replay buffer, and Q-learning agent. Currently developing unit tests for game environment and training loop with evaluation (`train.py` and `eval.py`). 

### Current Implementations ### 

- **`mappings.py`** Coordinate conversions between the (macro_square, local_square) pair and the (row, col) board positions) 
- **`env.py`** Consists of the implemented game rules, including a state array of the macro_squares and a 4-channel board state implementation 
- **`network.py`** CNN-based Q-network with 3 Conv2D layers, 2 fully-connected layers, and ReLU activations 
- **`memory.py`** Using a deque to store game transitions or "move plays" 
- **`agent.py`** A DQN agent with epsilon-greedy exploration strategy, action masking (for only legal actions), and the bellman equation for target calculation, MSE loss + backpropagation 

### Developer Notes ### 

Built and designed independently; worked with Claude for debugging support and validations. 
