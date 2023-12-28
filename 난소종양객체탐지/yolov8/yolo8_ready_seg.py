"""
Created on Oct 17, 2023

@author: jiyunHan (Wiseitech)

"""

# train :: nohup python /home/src/yolov8/yolo8_ready_seg.py --task train >  /home/src/log/SONO_mid_train_2.log 2>&1 &
# test :: python /home/src/yolov8/yolo8_ready_seg.py --task test --model_path /home/src/model/best_SONO_mid/weights/best.pt

# (CT) predict :: python /home/src/yolov8/yolo8_ready_seg.py --task predict --model_path /home/src/model/best_CT_mid/weights/best.pt --predict_path /home/data/best_CT_mid_yolodata/YOLODataset/images/test/10008_CTA_13.png
# (초음파) predict :: python /home/src/yolov8/yolo8_ready_seg.py --task predict --model_path /home/src/model/best_SONO_mid_sum/weights/best.pt --predict_path /home/data/best_SONO_mid_yolodata/YOLODataset/images/test/13284_SONO_2.png


import os
import time
import yaml
import argparse
import logging
import torch
from datetime import datetime
from ultralytics import YOLO
import warnings
warnings.filterwarnings(action='ignore')

import json
import pandas as pd
import numpy as np
import pycocotools.mask as mask
import cv2

os.chdir("/home/src/yolov8") # 위치 설정
os.getcwd()
pd.set_option('display.max_rows', None) # row 생략 없이 출력
pd.set_option('display.max_columns', None) # col 생략 없이 출력


def polygonFromMask(maskedArr):

    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
    # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError
    return segmentation
    
# 폴리곤 정보를 이미지 크기에 맞게 변환하는 함수
def scale_polygon(poly, image_width, image_height):
    scaled_poly = []
    for i in range(0, len(poly), 2):
        x = int(poly[i] * image_width)
        y = int(poly[i + 1] * image_height)
        scaled_poly.append((x, y))
    return np.array(scaled_poly, dtype=np.int32)

# 폴리곤과 클래스 라벨을 이미지에 그리는 함수
def draw_polygons_with_labels(image, polygons, class_labels):
    
    dir = '/'.join(args.predict_path.split("/")[0:5])
    with open(os.path.join(dir,'dataset.yaml'), encoding='UTF-8') as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    class_nm = _cfg['names']
    
    for poly, class_label in zip(polygons, class_labels):
        
        # 폴리곤 그리기
        scaled_poly = scale_polygon(poly, image.shape[1], image.shape[0])
        alpha = 0.4  # 투명도 조절
        overlay = image.copy()
        cv2.fillPoly(overlay, [scaled_poly], (0, 255, 0))
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # 바운딩 박스 그리기
        x, y, w, h = cv2.boundingRect(scaled_poly)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 클래스 라벨 텍스트 설정
        label_text = f"{class_nm[int(class_label)]}"

        # 클래스 라벨 텍스트 그리기
        cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

def draw_segmentation_labels(png_dir, json_dir, save_dir):
    
    with open (json_dir, "r", encoding = "utf8") as f:
        data = f.readlines()
    
    for rst in data:
        class_labels = [rst.split(" ")[0]]
        seg_boxes = rst.split(" ")[1:]
        
    boxes = [[float(value) for value in seg_boxes]]
    
    img_array = np.fromfile(png_dir, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image_height, image_width, _ = image.shape

    # 폴리곤과 클래스 라벨을 이미지에 그리기
    draw_polygons_with_labels(image, boxes, class_labels)
    
    cv2.imwrite(save_dir, image)



# set args
parser = argparse.ArgumentParser(description='yolov8 - object detection')

parser.add_argument(
    "--task",
    type=str,
    nargs="?",
    default="train",
    help="train or test or predict",
)

parser.add_argument(
    "--model_path",
    type=str,
    nargs="?",
    default=None,
    help="model path",
)

parser.add_argument(
    "--predict_path",
    type=str,
    nargs="?",
    default=None,
    help="predict path",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    nargs="?",
    default="/home/data/tmp_data",
    help="dataset path",
)

args = parser.parse_args()

start = time.time()

ori_yolo_data_path = "/home/data/tmp_data/YOLODataset"
yaml_path = os.path.join(ori_yolo_data_path,"dataset.yaml")

model_path = args.model_path

if args.task == 'train':
    
    # Load a model - segmentation
    model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch
    model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

    model.train(data=yaml_path, epochs=100, patience=30, batch=32, imgsz=512)
    path = model.export(format="onnx")  # export the model to ONNX format


elif args.task == 'test':
    
    # set log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S'
                                )
    file_handler = logging.FileHandler('/home/src/log/YOLO_TEST.log', mode='w') ## 파일 핸들러 생성
    file_handler.setFormatter(formatter) ## 텍스트 포맷 설정
    logger.addHandler(file_handler) ## 핸들러 등록

    logger.info(f'python /home/src/yolov8/yolo8_ready_seg.py --task {args.task} --model_path {args.model_path} --dataset_path {args.dataset_path}')
    
    logger.info(f'{args.task} - PROCESS_START')
    
    yolo_data_path = os.path.join(args.dataset_path,"YOLODataset")
    test_yaml_path = os.path.join(yolo_data_path,"dataset.yaml")
    
    
    # yaml 파일의 평가 데이터 셋을 임시로 test 폴더로 변경
    
    with open(test_yaml_path, "r") as f:
        read_data = yaml.safe_load(f)

    read_data['val'] = os.path.join(yolo_data_path,"images/test/")
    
    tmp_dir = os.path.join(yolo_data_path,'tmp.yaml')
    
    with open(tmp_dir, 'w') as p:
        yaml.dump(read_data, p)

    model = YOLO(model_path)
    
    metrics = model.val(model = model_path, data = tmp_dir, batch=8, imgsz=512, save_json= True)
    
    if os.path.isfile(tmp_dir):
        os.remove(tmp_dir)
    
    B_precison = metrics.results_dict["metrics/precision(B)"]
    B_recall = metrics.results_dict["metrics/precision(B)"]
    
    M_precison = metrics.results_dict["metrics/precision(M)"]
    M_recall = metrics.results_dict["metrics/precision(M)"]
    
    f1_score_B = 2 * ((B_precison*B_recall) /(B_precison+B_recall))
    f1_score_M = 2 * ((M_precison*M_recall) /(M_precison+M_recall))
    
    # 개별 결과값 로그 만들기 (폴리곤 값이 mask RLE 압축형식으로 기록되어있어 polygon 형식으로 변환)
    with open ("./runs/segment/val/predictions.json", "r") as f:
        data = json.load(f)
    
    df_rst = pd.DataFrame(data)
    
    copy_df_rst = df_rst.copy()
    
    for num, df_loop in enumerate(df_rst["segmentation"]):
        
        try:
            maskedArr = mask.decode(df_loop)
            
            RLE_to_polygon = polygonFromMask(maskedArr)
            
            copy_df_rst["segmentation"][num] = RLE_to_polygon
            
        except:
            copy_df_rst["segmentation"][num] = None
    

    logger.info(copy_df_rst)   
    
    logger.info(metrics)    
    logger.info(f"✅ 평가 결과 \n➡️ mAP : {metrics.seg.map50: .3f} \n\n➡️ precison : {M_precison: .3f} \n➡️ recall : {M_recall: .3f} \n➡️ f1-score : {f1_score_M: .3f}")
    
    # 모델 평가 결과 print
    print(f"✅ 평가 결과 \n➡️   mAP : {metrics.seg.map50: .3f}")
  
    logger.info("FINISH")

elif args.task == "predict":
    
    model = YOLO(model_path)
    result = model.predict(args.predict_path, save=True, save_conf=True)
    
    # 실제이미지에 라벨 그리기
    filenm, ext = os.path.splitext(args.predict_path)
    
    nm = filenm.split("/")[-1]
    path_front = filenm.split("/")[0:-3]
    lbl_path = os.path.join("/".join(path_front),"labels","test",f"{nm}.txt")
    
    save_dir = f"/home/src/yolov8/runs/segment/predict/ori_{nm}.png"
    
    draw_segmentation_labels(args.predict_path, lbl_path, save_dir)
    
    
else:
    print(" Error - Insert parameter --task { 'train' or 'test' or 'predict' } \
          \n example. python /home/yolo8_ready_seg.py --task test ")
    

print(f"time : {time.time() - start : .3f}")  # 현재시각 - 시작시간 = 실행 시간
