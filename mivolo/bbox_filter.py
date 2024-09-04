import json

# POI definido com (x, y, w, h) (288, 322, 770, 396) (288, 446, 738, 273) (262, 306, 470, 252)
poi_x = 402
poi_y = 406
poi_w = 532
poi_h = 314

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]

    iou = interArea / float(boxAArea)
    return iou

with open('C:/Users/urban/OneDrive/Documentos/GitHub/MiVOLO/jsons/Teste_Fila_6_6_20240630181728.avi.detections.json') as f:
    data_detection = json.load(f)

poi_box = [poi_x, poi_y, poi_w, poi_h]

# filtered_detections = [
#     d for d in data_detection
#         if d['bbox'][0] >= poi_x 
#         and d['bbox'][1] >= poi_y 
#         and d['bbox'][0] + d['bbox'][2] <= poi_x + poi_w
#         and d['bbox'][1] + d['bbox'][3] <= poi_y + poi_h
# ]

filtered_detections = [
    d for d in data_detection
    if calculate_iou(d['bbox'], poi_box) >= 0.6 
]


id_mapping = {}
new_id = 1

# novos ids
# for detection in filtered_detections:
#     old_id = detection["uniqueId"]
#     if old_id not in id_mapping:
#         id_mapping[old_id] = new_id
#         new_id += 1
#     detection["uniqueId"] = id_mapping[old_id]

with open('C:/Users/urban/OneDrive/Documentos/GitHub/MiVOLO/jsons/test_detectionsFinal0.6.json', 'w') as f:
    json.dump(filtered_detections, f, indent=4)

print(f"Total de detecções filtradas: {len(filtered_detections)}")
