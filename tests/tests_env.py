import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import UltimateTTTEnv
from mappings import to_global_idx

def test_row_win():
  """testing the row win logic for squares within the macrosquare, MARK macrosquare marked as won if true"""
  env = UltimateTTTEnv() 
  env.reset() 
  macro_square = 0 
  for local_square in range(3): 
    env.board[to_global_indx(macro_square, local_square)] = 1 # set these local squares to have being marked by the SAME player
  assert env.check_macro_win(macro_square) == 1 # should be marked 

def test_col_win():
  env = UltimateTTTEnv() 
  env.reset() 
  macro_square = 0
  for local_square in [0,3,6]:
    env.board[to_global_indx(macro_square, local_square)] = 1 # set the local individiaul squares accessed within the entire 81-square board to 1
  assert env.check_macro_win(macro_square) == 1 # should be marked 

try: 
  test_row_win() 
  print("Passed test 1")
  test_col_win()
  print("Passed test 2") 
except: 
  print("ERROR") 
  


