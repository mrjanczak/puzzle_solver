import os
import sys

local = os.path.dirname(__file__)
os.chdir(local)
print(f'Current path {local}')

parent_dir = os.path.abspath(os.path.join(local, '..'))
sys.path.append(parent_dir)
from solver import *

# Tests
process_pieces(p_set = 100, pieces_dir = os.path.join(local, 'cornerBR'), files_range = [1,1], plot_piece=True, locks_range=[2,3])
exit()

# puzzle size
M, N = 36, 28

# My convention used for Vermer - not recommended
# p_i_start = 1
# p_set_mult = 1000   
# p_id_offset = 10 # piece_id printed in terminal (in puzzle[] id has no offset) - I marked my real pieces with id starting from 11 to 110 within each set

# Recommended convention
p_i_start = 0
p_set_mult = 100 # puzzle table in .md format is narrower   
p_id_offset = 0 # id in puzzle[] and on real puzzle are consistent - mark your pieces from 0 to 99 within each set

# Step 1. Process all pieces and save (You can also add some pieces later in subsequent sets)
process_pieces(p_set = 0, pieces_dir = os.path.join(local, 'col1'),   locks_range=[2,3], p_i_start = p_i_start, p_set_mult = p_set_mult)
process_pieces(p_set = 1, pieces_dir = os.path.join(local, 'row1'),   locks_range=[2,3], p_i_start = p_i_start, p_set_mult = p_set_mult)  
process_pieces(p_set = 2, pieces_dir = os.path.join(local, 'colM'),   locks_range=[2,3], p_i_start = p_i_start, p_set_mult = p_set_mult)
process_pieces(p_set = 3, pieces_dir = os.path.join(local, 'corBR'),  locks_range=[2,3], p_i_start = p_i_start, p_set_mult = p_set_mult) # actual set 50 in .pkl file

process_pieces(p_set = 10, pieces_dir = os.path.join(local, 'int10'), locks_range=[4,4], p_i_start = p_i_start, p_set_mult = p_set_mult, files_range = [1,100])
process_pieces(p_set = 20, pieces_dir = os.path.join(local, 'int20'), locks_range=[4,4], p_i_start = p_i_start, p_set_mult = p_set_mult, files_range = [1,100])
process_pieces(p_set = 30, pieces_dir = os.path.join(local, 'int30'), locks_range=[4,4], p_i_start = p_i_start, p_set_mult = p_set_mult, files_range = [1,100])
process_pieces(p_set = 40, pieces_dir = os.path.join(local, 'int40'), locks_range=[4,4], p_i_start = p_i_start, p_set_mult = p_set_mult, files_range = [1,11] )

# Step 2. Load needed pieces
pieces = {}
pieces = load_pieces(p_set = 0, pieces = pieces)
pieces = load_pieces(p_set = 1, pieces = pieces)
pieces = load_pieces(p_set = 2, pieces = pieces)
pieces = load_pieces(p_set = 3, pieces = pieces)

pieces = load_pieces(p_set = 10, pieces = pieces)
pieces = load_pieces(p_set = 20, pieces = pieces)
pieces = load_pieces(p_set = 30, pieces = pieces)
pieces = load_pieces(p_set = 40, pieces = pieces)

# Step 3. Initiate puzzle
puzzle = init_puzzle(puzzle = (M, N), pieces = pieces, p_file = 'IMG_20250322_165940', p_pos = (1,N), p_edge = 4) 
puzzle = init_puzzle(puzzle = puzzle, pieces = pieces, p_file = 'IMG_20250327_230157', p_pos = (M,1), p_edge = 2) 

# Step 4. Match frame
puzzle = match_puzzle(puzzle, pieces, from_pos=(1,N), to_pos=(M,N),      p_id_offset = p_id_offset )
puzzle = match_puzzle(puzzle, pieces, from_pos=(1,N), to_pos=(1,1),      p_id_offset = p_id_offset )
puzzle = match_puzzle(puzzle, pieces, from_pos=(1,1), to_pos=(23,1),     p_id_offset = p_id_offset )
puzzle = match_puzzle(puzzle, pieces, from_pos=(M-1, 2), to_pos=(M-8,3), p_id_offset = p_id_offset )

# Step 5. Match internal pieces

# Read puzzle
puzzle = read_puzzle()

# Update puzzle manually if needed
# puzzle = set_puzzle(puzzle, 36, 1, -1)

puzzle = match_puzzle(puzzle, pieces, from_pos=(M-1, 2), to_pos=(M-8,3), verbose=False, p_id_offset = p_id_offset )


