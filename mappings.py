SQUARE_TO_IDX = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'K':8}
CELL_TO_SQUARE = {SQUARE_TO_IDX[key] : key for key in SQUARE_TO_IDX}

def cell_to_next_square(cell_number):
    """helper function for routing, mapping cell to active square """
    return CELL_TO_SQUARE[cell_number]

def to_global_idx(macro_square,idx):
    """convert from 2D coordinates to 1x81 one-D vector"""
    # same conversions to convert from row/col to macro-square/index
    row = (macro_square // 3)*3 + idx // 3
    col = (macro_square % 3)*3 + idx % 3
    return row * 9 + col

def to_local_coords(global_idx):
    """convert from element of 1D vector to 2D coordinate in 9x9 grid"""
    row = global_idx // 9 # row
    col = global_idx % 9 # column
    macro_square = (row // 3)*3 + col // 3 # big square
    cell_number = (row % 3)*3 + col % 3 # individual cells
    return (macro_square, cell_number) # return as a tuple coordinate pair




