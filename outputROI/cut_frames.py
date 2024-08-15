import cv2
import os
import shutil

# Caminho para o arquivo de vídeo
file_path = 'C:/Users/urban/OneDrive/Documentos/GitHub/MiVOLO/outputROI/out_Teste Fila_6_6_20240630181728Frente(MV1256body_ROI).avi'
cap = cv2.VideoCapture(file_path)

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {file_path}")
    exit()

# Extrair o nome do arquivo de vídeo sem a extensão
video_name = os.path.splitext(os.path.basename(file_path))[0]

# Pasta para salvar os frames extraídos
dest_path = 'frames_to_see'
frames_to_skip = 100

# Remover e recriar o diretório de destino se já existir
if os.path.exists(dest_path):
    shutil.rmtree(dest_path)
os.makedirs(dest_path)

ct = 0

# Loop para ler os frames e salvá-los a cada 100 frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if ct % frames_to_skip != 0:
        ct += 1
        continue
    
    # Salvar o frame com o nome do vídeo incluído
    output_frame_path = f'{dest_path}/{video_name}_frame_{ct}.jpg'
    success = cv2.imwrite(output_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    if success:
        print(f'Frame {ct} salvo em {output_frame_path}')
    else:
        print(f'Erro ao salvar frame {ct}')
    
    ct += 1

cap.release()
print("Processamento de frames concluído.")
