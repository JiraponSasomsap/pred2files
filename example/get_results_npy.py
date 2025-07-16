import sys
sys.path.append('..')

from src.get_results import GetNPy

getnp = GetNPy('results-npy',max_buffer_len=10)
for i in range(1, 9):
    print(getnp.get_result_by_iframe(i))