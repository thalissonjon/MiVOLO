from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
from typing import Dict, Generator, List, Optional, Tuple

import json
import os
import cv2
import numpy as np

import logging

def configure_logger(name, logger_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logger_level)
    
    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logger_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to ch
    ch.setFormatter(formatter)
    
    # Add ch to logger
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger

logger = configure_logger(__name__)

class AGDetections:
    # checkpoint = mivolo
    # detector_weights = yolo face/person detector 
    def __init__(self, checkpoint, detector_weights, device="cuda", verbose=True, logger=None):
        if logger:
            self.logger = logger 
        else:
            self.logger = configure_logger(__name__)
        
        self.logger.info(f'Iniciando modelo de detecção {detector_weights}')
        self.associated_persons_ids = []
        self.detector = Detector(detector_weights, device, verbose)
        self.logger.info(f'Iniciando modelo MiVOLO {checkpoint}')
        self.age_gender_model = MiVOLO(
            checkpoint,
            device,
            half=True,
            use_persons=True,
            disable_faces=False,
            verbose=verbose,
        )

    def detect_age_gender(self, img, frame_detections):
        
        detected_objects, out_im = self.recognize(img)

        persons = self.extract_person_info(detected_objects)

        # for person in persons:
        #     print(f"ID: {person['id']}, BBox: {person['bbox']}, Age: {person['age']}, Gender: {person['gender']}")
        
        updated_json = self.associate_and_update_json(frame_detections, persons)
        self.logger.debug(updated_json)

        with open('output/jsonAtt.json', 'w') as f:
            json.dump(updated_json, f, indent=4)


        # bname = os.path.splitext(os.path.basename(img))[0]
        filename = os.path.join('outputFrame', f"out_test.jpg")
        cv2.imwrite(filename, out_im)
        print(f"Saved result to {filename}")
    

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.track(image)
        # detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        out_im = detected_objects.plot()

        return detected_objects, out_im


    def calculate_iou(self, bbox1, bbox2):
        # x y w h
        x1, y1, w1, h1 = bbox1
        b_x1, b_y1, b_x2, b_y2 = bbox2

        # calcular x e y da interseçao das boxes
        inter_x1 = max(x1, b_x1)
        inter_y1 = max(y1, b_y1)
        # inter_x2 = min(x1 + w1, b_x1 + b_x2)
        # inter_y2 = min(y1 + h1, b_y1 + b_y2)
        inter_x2 = min(x1 + w1, b_x2)
        inter_y2 = min(y1 + h1, b_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        bbox1_area = w1 * h1
        bbox2_area = (b_x2 - b_x1) * (b_y2 - b_y1)

        iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
        
        return iou


    # colocar recebimento detecções e detecções realizadas numero detecçoes ja associadas
    def associate_and_update_json(self, json_data, persons, iou_threshold=0.8):  
        for entry in json_data: # detecçoes ds
            json_id = entry['uniqueId']

            if json_id in self.associated_persons_ids:
                print(f'{json_id} já foi detectado')
                continue 
        
            json_bbox = entry['bbox']
            best_iou = 0 
            best_person = None

            for person in persons: # detecçoes yolo
                # person_id = person['id']
                person_bbox = person['bbox']
                iou = self.calculate_iou(json_bbox, person_bbox)
                # print(f"IOU entre {person_id}{person_bbox} e {json_id.data}{json_bbox}: {iou}") 
                
                if iou > best_iou:
                    best_iou = iou
                    best_person = person
            
            

            if best_iou > iou_threshold and best_person is not None:
                entry['age'] = best_person['age']
                entry['gender'] = best_person['gender']
                self.associated_persons_ids.append(json_id)
                # print(self.associated_persons_ids)
            else:
                self.logger.info(f"nenhuma associaçao adequada encontrada para {json_bbox}")
    
        return json_data
    
    def extract_person_info(self, detected_objects):
        person_info = []
        person_inds = detected_objects.get_bboxes_inds("person")

        for ind in person_inds:
            person_id = detected_objects._get_id_by_ind(ind)
            bbox = detected_objects.get_bbox_by_ind(ind)
            age = detected_objects.ages[ind]
            gender = detected_objects.genders[ind]

            person_info.append({
                "id": person_id,
                "bbox": bbox.cpu().numpy().tolist(),
                "age": age,
                "gender": gender
            })

        return person_info
    



    
