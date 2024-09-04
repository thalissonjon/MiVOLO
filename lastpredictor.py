from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
import torch
from ultralytics.engine.results import Results, Boxes
import json

class Predictor:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw
        self.json_path = config.json_path
        self.body_bboxes = self.load_body_bboxes(self.json_path)
    
    def load_body_bboxes(self, json_path: str) -> Dict[int, List[Dict[str, Union[int, List[int]]]]]:
        with open(json_path, 'r') as f:
            body_bboxes = json.load(f)
        body_bboxes_by_frame = defaultdict(list)
        for item in body_bboxes:
            frame_number = item["frame_number"]
            body_bboxes_by_frame[frame_number].append(item)
        return body_bboxes_by_frame
    

    def create_results_from_json(self, body_bboxes: List[Dict[str, Union[int, List[int]]]], image_shape: Tuple[int, int]):
        boxes = []
        for bbox in body_bboxes:
            x1, y1, w, h = bbox['bbox']
            x2, y2 = x1 + w, y1 + h
            confidence = 1.0  # confiança ficticia
            class_id = 0  # 0 = pessoa
            track_id = bbox.get('uniqueId', -1)  # id de tracking p usar na face tb

            # adiciona os dados de cada bounding box ao tensor
            boxes.append([x1, y1, x2, y2, confidence, track_id, class_id])

        # list to tensor pyt
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device='cuda')

        # verificar dim pq tava dando erro antes (verificar origem)
        if boxes_tensor.ndim == 1:
            boxes_tensor = boxes_tensor.unsqueeze(0)

        # criar o obj Results
        results = Results(
            orig_img=np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8),
            path='image0.jpg', # ?
            names={0: 'person'},
            boxes=boxes_tensor
        )

        return results



    def recognize(self, image: np.ndarray, frame_number: int) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        
        # Pegar as bbox do corpo pro frame atual, se existir
        body_bboxes = self.body_bboxes.get(frame_number, [])
        if body_bboxes:
            json_results = self.create_results_from_json(body_bboxes, image.shape[:2])
            detected_objects.merge_results(json_results)

        self.age_gender_model.predict(image, detected_objects)
        out_im = None
        if self.draw:
            out_im = detected_objects.plot()
        return detected_objects, out_im

    def recognize_video(self, source: str, roi=None) -> Generator:
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_number in tqdm.tqdm(range(total_frames)):
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to read the frame")
                break
            
            if roi:
                x, y, w, h = roi
                frame_cropped = frame[y:y+h, x:x+w]
            else:
                frame_cropped = frame

            detected_objects, out_im = self.recognize(frame_cropped, frame_number)
            
            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            for guid, data in cur_persons.items():
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            detected_objects.set_tracked_age_gender(detected_objects_history)

            if self.draw:
                try:
                    annotated_cropped_frame = out_im if out_im is not None else frame_cropped
                    if roi:
                        frame[y:y+h, x:x+w] = annotated_cropped_frame
                    else:
                        frame = annotated_cropped_frame
                except ValueError as e:
                    print(f"Error drawing on the cropped frame: {e}")
                    continue

                if roi:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            yield detected_objects_history, frame
