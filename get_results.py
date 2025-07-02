from src.get_results import GetResultsNumpy
from pathlib import Path
import cv2

FILES = Path(R'C:\Users\JiraponSasomsap\ArticulusProjects\tools\pred2files\D01_20250529140126_re-encode\results-npy')
VIDEO_PATH = Path(R"C:\Users\JiraponSasomsap\ArticulusProjects\projects\PTG\punthai_videos\29\D01_20250529140126_re-encode.mp4")

cap = cv2.VideoCapture(str(VIDEO_PATH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

result = GetResultsNumpy(pred_files=FILES, max_buffer_len=50)

np_results = result.get_results(iframe=1, frame=cap.read()[1])

print("Bounding Boxes      :", np_results.bounding_boxes)
print("Bounding Boxes Norm :", np_results.bounding_boxes_norm)
print("Center              :", np_results.center)
print("Center Norm         :", np_results.center_norm)
