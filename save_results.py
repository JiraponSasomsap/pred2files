from ultralytics import YOLO
from src.save_results import SaveNumpy
import cv2
from pathlib import Path
from submodules import cv3

VIDEO_PATH = Path(R"C:\Users\JiraponSasomsap\ArticulusProjects\projects\PTG\punthai_videos\29\D01_20250529140126_re-encode.mp4")
# VIDEO_PATH = Path(R"C:\Users\JiraponSasomsap\ArticulusProjects\tools\pred2files\submodules\cv3\assets\hello_opencv.mp4")
MODEL_PATH = Path(R"C:\Users\JiraponSasomsap\ArticulusProjects\public_models\yolov8n.pt")
SAVE_AS = Path(R".")

model = YOLO(model=MODEL_PATH)
model_config = {
    'verbose': False,
}

save_numpy = SaveNumpy(
    save_as=SAVE_AS,
    max_buffer_len=50
)

cap = cv2.VideoCapture(VIDEO_PATH)

save_numpy.set_config(
    model_path=MODEL_PATH,
    video_path=VIDEO_PATH,
    fps=round(cap.get(cv2.CAP_PROP_FPS)),
    max_frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
)

saved_path = None

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Prepare data to save
        result = model.predict(frame, **model_config)[0]
        boxes = result.boxes.xyxyn.cpu().numpy()

        # Save results to numpy
        saved_path = save_numpy.npy(arr=boxes)

        # Display the results
        if cv3.imshow(result.plot()).interrupt(1):
            break
        # break
finally:
    cap.release()
    cv3.destroyAllWindows()

    if saved_path:
        save_numpy.config['predictor']['predictor_params'] = model_config
        save_numpy.save_config(save_as=saved_path).dump_npy()