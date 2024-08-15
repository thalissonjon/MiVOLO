from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult


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

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_objects, out_im

    def recognize_video(self, source: str, roi = None) -> Generator:

        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm.tqdm(range(total_frames)):
            ret, frame = video_capture.read()
            if not ret:
                print("falha ao ler o frame")
                break
            
            if roi:
                x, y, w, h = roi
                frame_cropped = frame[y:y+h, x:x+w]
            else:
                frame_cropped = frame

            # detected_objects: PersonAndFaceResult = self.detector.track(frame)
            detected_objects: PersonAndFaceResult = self.detector.track(frame_cropped)
            # self.age_gender_model.predict(frame, detected_objects)
            self.age_gender_model.predict(frame_cropped, detected_objects)

            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            # add tr_persons and tr_faces to history
            for guid, data in cur_persons.items():
                # not useful for tracking :)
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            detected_objects.set_tracked_age_gender(detected_objects_history)
            # if self.draw:
            #     frame = detected_objects.plot()
            #     # frame_cropped = detected_objects.plot(frame_cropped)
            #     # detected_objects.plot(frame)

            if self.draw:
                try:
                    annotated_cropped_frame = detected_objects.plot()
                    if roi:
                        frame[y:y+h, x:x+w] = annotated_cropped_frame
                    else:
                        frame = annotated_cropped_frame
                except ValueError as e:
                    print(f"erro ao desenhar no frame_cropped: {e}")
                    continue

            # desenhar roi no frame orig
                if roi:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            yield detected_objects_history, frame
