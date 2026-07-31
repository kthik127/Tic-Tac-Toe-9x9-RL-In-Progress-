SQUARE_TO_IDX = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'K':8}
CELL_TO_SQUARE = {SQUARE_TO_IDX[key] : key for key in SQUARE_TO_IDX}

def cell_to_next_square(cell_number):
    """helper function for routing, mapping cell to active square """
    return CELL_TO_SQUARE[cell_number]

def to_global_idx(macro_square,idx):
    """convert from 2D coordinates to 1x81 one-D vector"""
    return SQUARE_TO_IDX[macro_square] * 9 + idx

def to_local_coords(global_idx):
    """convert from element of 1D vector to 2D coordinate in 9x9 grid"""
    macro_square = CELL_TO_SQUARE[global_idx // 9]
    cell_number = global_idx % 9
    return (macro_square, cell_number)



