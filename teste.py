from AG_detection import AGDetections
import time
import cv2
import json
import os

mivolo_path = 'models/model_imdb_cross_person_4.22_99.46.pth.tar'
detector_path_personface = 'models/yolov8x_person_face.pt'
detector_path_face = 'models/yolov8l-face.pt' 
detector_path_person = 'models/yolov8n_custom_1280.pt'
all_detections = 'jsons/test_detectionsFinal0.6.json'

frame_path = 'videos/Teste Fila_6_6_20240630181728_frame1454.png'
ds_detections = 'jsons/Teste_Fila_6_6_20240630181728.avi.last_detections.json'

frame2_path = 'videos/Teste Fila_6_6_20240630181728_frame1453.png'
ds_detections2 = 'jsons/last_detections1453.json'

ds_detectionsPOI2 = 'jsons/poi2.json'

detections = AGDetections(mivolo_path, detector_path_person, detector_path_face)
img = cv2.imread(frame_path)
poi = (402, 406, 532, 314)

with open(ds_detectionsPOI2, 'r') as f:
    json_data = json.load(f)

# while True:
#     detections.detect_age_gender(img, json_data)
#     time.sleep(10)

count = 0

# img_path = 'frames_input/frame_0003.png'
# img = cv2.imread(img_path)
# frame_detections = [det for det in json_data if det["frame_number"] == 3]
# detections.detect_age_gender(img, frame_detections)

for image_file in os.listdir('frames_input'):
    img_path = os.path.join('frames_input', image_file)
    frame_detections = [det for det in json_data if det["frame_number"] == count]
    img = cv2.imread(img_path)
    detections.detect_age_gender(img, frame_detections)
    # detections.detect_age_gender(img, json_data)
    count +=1


detections.generate_csv()

    