import json

# Definir o ROI
roi_x_min = 288
roi_y_min = 322
roi_x_max = 770
roi_y_max = 396

with open('path/to/your_file.json') as f:
    data_detection = json.load(f)

# Filtrar todas as detecções no JSON para cada frame
filtered_detections = [
    d for d in data_detection
    if d['bbox'][0] >= roi_x_min 
    and d['bbox'][1] >= roi_y_min 
    and d['bbox'][0] + d['bbox'][2] <= roi_x_max
    and d['bbox'][1] + d['bbox'][3] <= roi_y_max
]

# Exibir as detecções filtradas
print(json.dumps(filtered_detections, indent=4))
