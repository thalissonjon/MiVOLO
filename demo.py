import argparse
import logging
import os

import cv2
import torch
import yt_dlp
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

_logger = logging.getLogger("inference")


def get_direct_video_url(video_url):
    ydl_opts = {
        "format": "bestvideo",
        "quiet": True,  # Suppress terminal output (remove this line if you want to see the log)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)

        if "url" in info_dict:
            direct_url = info_dict["url"]
            resolution = (info_dict["width"], info_dict["height"])
            fps = info_dict["fps"]
            yid = info_dict["id"]
            return direct_url, resolution, fps, yid

    return None, None, None, None


def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source {vid_uri}")
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return res, fps


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--output", type=str, default=None, required=True, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")

    return parser

def select_roi(image):
    roi = cv2.selectROI("Selecione a ROI e pressione Enter", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)

    input_type = get_input_type(args.input)

    if input_type == InputType.Video or input_type == InputType.VideoStream:
        if not args.draw:
            raise ValueError("Video processing is only supported with --draw flag. No other way to visualize results.")

        if "youtube" in args.input:
            args.input, res, fps, yid = get_direct_video_url(args.input)
            if not args.input:
                raise ValueError(f"Failed to get direct video url {args.input}")
            outfilename = os.path.join(args.output, f"out_{yid}.avi")
        else:
            bname = os.path.splitext(os.path.basename(args.input))[0]
            outfilename = os.path.join(args.output, f"out_{bname}.avi")
            res, fps = get_local_video_info(args.input)

        cap = cv2.VideoCapture(args.input)
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("nao foi possivel pegar o primeiro frame do video")
        
        #roi
        # roi = select_roi(first_frame)
        roi = (288, 322, 770, 396)

        print(f"ROI selecioando: {roi}")
        if args.draw:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(outfilename, fourcc, fps, res)
            _logger.info(f"Saving result to {outfilename}..")

        for (detected_objects_history, frame) in predictor.recognize_video(args.input, roi=roi):

            if args.draw:
                out.write(frame)
                # out.write(frame_cropped)

    elif input_type == InputType.Image:
        image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]

        for img_p in image_files:

            img = cv2.imread(img_p)
            detected_objects, out_im = predictor.recognize(img)

            if args.draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(args.output, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                _logger.info(f"Saved result to {filename}")


if __name__ == "__main__":
    main()

# IMDB DATASET
#python demo.py --input "videos/catraca_1MTest.avi" --output "output" --detector-weights "models/yolov8x_person_face.pt " --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --with-persons --draw  #MIVOLOD1 + YOLOV8XFACEPERSON + body e face output\out_catraca_1MTest(MV1256face_body).avi
#python demo.py --input "videos/catraca_1MTest.avi" --output "output" --detector-weights "models/yolov8x_person_face.pt " --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --with-persons --disable-faces --draw # MIVOLOD1+ YOLOV8FACEPERSON + apenas body output\out_catraca_1MTest(MV1256body).avi
#python demo.py --input "videos/catraca_1MTest.avi" --output "output" --detector-weights "models/yolov8x_person_face.pt " --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --draw # MIVOLOD1+ YOLOV8FACEPERSON + apenas face output\out_catraca_1MTest(MV1256face).avi
#python demo.py --input "videos/catraca_1MTest.avi" --output "output" --detector-weights "models/yolov8l-face.pt " --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --draw # MIVOLOD1+ YOLOV8FACEPERSON + apenas face output\out_catraca_1MTest(MV1256face_yolov8lface).avi

#python demo.py --input "videos/Teste Fila_6_6_20240629231343.avi" --output "output" --detector-weights "models/yolov8x_person_face.pt" --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --with-persons --draw  #MIVOLOD1 + YOLOV8XFACEPERSON + body e face output\out_Teste Fila_6_6_20240629231343(MV1256face_body).avi

#python demo.py --input "videos/Teste Fila_6_6_20240630181728Frente.avi" --output "output" --detector-weights "models/yolov8x_person_face.pt" --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --with-persons --draw #MIVOLOD1 + YOLOV8XFACEPERSON + body e face output\out_Teste Fila_6_6_20240630181728Frente(MV1256face_body).avi
#python demo.py --input "videos/Teste Fila_6_6_20240630181728Frente.avi" --output "output" --detector-weights "models/yolov8x_person_face.pt" --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --draw #MIVOLOD1 + YOLOV8XFACEPERSON + apenas face output\out_Teste Fila_6_6_20240630181728Frente(MV1256face).avi
#python demo.py --input "videos/Teste Fila_6_6_20240630181728Frente.avi" --output "output" --detector-weights "models/yolov8l-face.pt" --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --draw #MIVOLOD1 + YOLOV8lface + apenas face output\out_Teste Fila_6_6_20240630181728Frente(MV1256face_yolov8lface
# 






#python demo.py --input "videos/catraca_1MTest.avi" --output "output" --detector-weights "models/yolov8x_person_face.pt " --checkpoint "models/d5_448_87.0.pth.tar" --device "cuda:0" --with-persons --draw  # MIVOLOD5 + YOLOV8XFACEPERSON + body e face       
# testcmd python demo.py --input "videos/catraca_1MTest.avi" --output "output" --detector-weights "models/yolov8x_person_face.pt " --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --draw # MIVOLOD1+ YOLOV8FACEPERSON output\out_catraca_1MTest(MV1256face).avi


# UTK DATASET
#python demo.py --input "videos/catraca_1MTest.avi" --output "output" --detector-weights "models/yolov8l-face.pt " --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --draw #MIVOLOD1 + YOLOV8lface + apenas face output + UTK DATASETMODEL \***************.avi



# COM ROI
# python demo.py --input "videos/Teste Fila_6_6_20240630181728Frente.avi" --output "outputROI" --detector-weights "models/yolov8l-face.pt" --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --draw   outputROI\out_Teste Fila_6_6_20240630181728Frente(MV1256face_yolov8lface_ROI).avi face apenas
# python demo.py --input "videos/Teste Fila_6_6_20240630181728Frente.avi" --output "outputROI" --detector-weights "models/yolov8x_person_face.pt" --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --draw --with-persons outputROI\out_Teste Fila_6_6_20240630181728Frente(MV1256face_body_ROI).avi body e face
# python demo.py --input "videos/Teste Fila_6_6_20240630181728Frente.avi" --output "outputROI" --detector-weights "models/yolov8x_person_face.pt " --checkpoint "models/model_imdb_cross_person_4.22_99.46.pth.tar" --device "cuda:0" --with-persons --disable-faces --draw outputROI\out_Teste Fila_6_6_20240630181728Frente(MV1256body_ROI).avi body apenas