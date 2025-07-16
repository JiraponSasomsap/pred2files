import sys
sys.path.append('..')
from submodules import cv3

from pathlib import Path
from src.save_results import SaveFileNPz
from ultralytics import YOLO

images_path = Path(r'C:\Users\JiraponSasomsap\ArticulusProjects\projects\TEENOI\extract-tn\JUS\images')
model = YOLO(r"C:\Users\JiraponSasomsap\ArticulusProjects\public_models\yolov8n-seg.pt")

npsave = SaveFileNPz(
    save_as='.',
    name='name',
    model_name_stem='v1',
    max_buffer_len=10
)

for im in images_path.glob('*.jpg'):
    result = model.predict(im)[0]
    datadict = {
        'boxes':result.boxes.cpu().data.numpy(),
        'seg':result.masks.cpu().xy
    }
    npsave.save_npz(data_dict=datadict)
    if cv3.imshow(result.plot()).interrupt(1):
        break
npsave.dump_npz()