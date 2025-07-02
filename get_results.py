from src.get_results import GetResultsNumpy
from pathlib import Path

FILES = Path(R'C:\Users\JiraponSasomsap\ArticulusProjects\tools\pred2files\D01_20250529140126_re-encode\results-npy')
VIDEO_PATH = Path(R"C:\Users\JiraponSasomsap\ArticulusProjects\projects\PTG\punthai_videos\29\D01_20250529140126_re-encode.mp4")

result = GetResultsNumpy(pred_files=FILES, max_buffer_len=50)

boxes = result.get_results(iframe=1)

print(boxes)