import torch
import numpy as np
from mappings import to_local_coords, to_global_idx


class UltimateTTTEnv:
    # egocentric design
    def __init__(self):
        """initializer of the environment"""
        self.board = torch.zeros(81, dtype = torch.int8) # entire coordinate list
        self.macro_status = torch.zeros(9, dtype = torch.int8) # status of 9 squares, 0 for active/not decided, 1 for player win, -1 for player loss, 2 for draw
        self.current_player = 1 # yes on current player, -1 means its the other player
        self.active_macro = None # macro square that player must go in, integer or None
        self.win_lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]] # winning moves

    def reset(self):
        """resets the game and board states, returns state"""
        self.board.fill_(0) # fill it with zeros
        self.macro_status.fill_(0) # fill it with zeros (all undecided now)
        self.current_player = 1
        self.active_macro = None # no active macro squares
        return self.encode_state() # return the game state as a tensor

    def check_macro_win(self, macro_idx):
        """checks if a macro_square has a winning combination of moves by a player or is in a draw, returns status"""
        start = macro_idx * 9
        macro_squares = self.board[start:start+9] # looking at 9 mini cells within a single macroboard

        for [c1,c2,c3] in self.win_lines:
            if macro_squares[c1] == macro_squares[c2] == macro_squares[c3] != 0: # if all are filled and NONzero (must be EQUAL though)
                return macro_squares[c1].item() # get the actual VALUE of the player that put it there
        if 0 not in macro_squares: # no wins but all squares are full
            return 2 # draw
        else:
            return 0 # undecided

    def legal_moves(self):
        """returns a list of all the legal moves for a particular gamestate"""
        if self.active_macro is None or self.macro_status[self.active_macro] != 0: # free choice OR macro square is already won
            return [cell for cell in range(len(self.board)) if self.board[cell] == 0 and self.macro_status[to_local_coords(cell)[0]] == 0] # any move in board thats not taken
        else:
            macro_squares = self.board[self.active_macro*9:self.active_macro*9+9] # only look within the specific macrosquare
            return [to_global_idx(self.active_macro,cell) for cell in range(len(macro_squares)) if macro_squares[cell] == 0] # only for those macrosquares that have open spots

    def step(self, action):
        """processing a move and checking win condition after each move"""
        reward = 0
        done = False
        cell = action.item()

        if cell not in self.legal_moves(): # first make sure the move is legal
            raise Exception("not a legal move.") # do not reward non-legal moves

        macro_square, next_square = to_local_coords(cell) # individual cell MAPS to CURRENT macro square number AND mini cell number, which is the next square
        self.board[cell] = self.current_player  # mark the cell

        # check if a current square is won

        result = self.check_macro_win(macro_square)
        self.macro_status[macro_square] = result

        # set the NEXT macro square as the new active square that the player must route too

        if self.macro_status[next_square] != 0: # already won or decided
            self.active_macro = None # free choice
        else:
            self.active_macro = next_square  # set the active square to the NEXT square

        # check win logic

        win_count = 0
        draw_count = 0

        if 0 not in self.macro_status or len(self.legal_moves()) == 0: # all macro squares are decided
            done = True # game is done
            for square in self.macro_status:
                if square == self.current_player: # count the number of squares player won
                    win_count += 1
                elif square == 2:
                    draw_count += 1 # count number of squares drawn
            opp_count = 9-draw_count-win_count # check winners
            if win_count < opp_count:
                reward = -1 # negative reward for loss
            elif win_count > opp_count:
                reward = 1 # positive reward for win

        self.current_player = -self.current_player # switch players

        return (self.encode_state(), reward, done, {}) # (next_state, reward, finished_condition, info)




    def encode_state(self):
        pass








