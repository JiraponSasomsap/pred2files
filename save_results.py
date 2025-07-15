from ultralytics import YOLO
from src.save_results import SaveNumpy
import cv2
from pathlib import Path
from submodules import cv3

VIDEO_PATH = Path(R"C:\Users\JiraponSasomsap\ArticulusProjects\projects\PTG\punthai_videos\29\D01_20250529140126_re-encode.mp4")
MODEL_PATH = Path(R"C:\Users\JiraponSasomsap\ArticulusProjects\public_models\yolov8n.pt")
SAVE_AS = Path(R".")

model = YOLO(model=MODEL_PATH)

save_numpy = SaveNumpy(
    save_as=SAVE_AS,
    name='name',
    model_name_stem='model_name_stem',
    project='project',
    max_buffer_len=50
)

cap = cv2.VideoCapture(VIDEO_PATH)
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    save_numpy.save_npz({'d':[count], 'dd':[]})
    count = count+1
    if cv3.imshow(frame).interrupt(1):
        break
    
save_numpy.dump_npy()
cv2.destroyAllWindows()