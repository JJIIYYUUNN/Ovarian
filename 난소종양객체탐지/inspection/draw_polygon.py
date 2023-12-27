import os
import cv2
import numpy as np
import json


# 의미정확성 FAIL 이미지 확인

import pandas as pd
import shutil

pd_data = pd.read_excel('/home/src/inspection/(036-049) 의미정확성 검사결과서_난소암 데이터_20230919.xlsx', sheet_name="TC2 검사이력 ", header=14, usecols = ["파일명", "검사 결과"])
data = pd_data.dropna(axis =0, how='any')
Fail_data = data[data["검사 결과"] == "PASS"]["파일명"]

# 의미정확성 FAIL 이미지 저장
   
data_type = "초음파"
label_input_path = "/home/data/first_data/2.라벨링데이터"
png_input_path = "/home/data/first_data/1.원천데이터"

save_path = "/home/data/tmp_data"

CT_folder_nm = '복부'

for ca in ["암","양성종양"]:
    
    if data_type == "CT":
    
        root_path = f"{label_input_path}/{ca}/{data_type}/{CT_folder_nm}/"
        root_imagepath = f"{png_input_path}/{ca}/{data_type}/{CT_folder_nm}"
    
    elif data_type == "초음파":
        root_path = f"{label_input_path}/{ca}/{data_type}/"
        root_imagepath = f"{png_input_path}/{ca}/{data_type}"

    file_list = os.listdir(root_path)
    img_file_list = os.listdir(root_imagepath)

    for label_file, img_file in zip(sorted(file_list), sorted(img_file_list)):
        
        nm, ext = os.path.splitext(label_file)
        
        if nm in Fail_data.to_list():
            label_path = os.path.join(root_path, label_file)
            image_path = os.path.join(root_imagepath, img_file)
            
            rgb_image = cv2.imread(image_path)

            with open (label_path, "r", encoding = "utf8") as f:
                data = json.load(f)

            filename = data["fileName"]
            taskname = data["taskName"]
            
            resultData = data["resultData"]
            

            for num, detail in enumerate(resultData):
                
                label = resultData[num]['value']
                poly = resultData[num]['points']
                tumor_type = resultData[num]['value']
                
                points_lst = []
                for pp in poly:
                    x_point = pp["x"]
                    y_point = pp["y"]
                    points_lst.append([x_point, y_point])
                
                
                pts = np.array(points_lst, dtype = np.int32)
                
                draw_polygon = cv2.polylines(rgb_image, [pts], True, (0,0,255), 2)
            
            folder_dir = os.path.join("/home/data/draw_polygon_success",taskname,data_type,tumor_type)
            
            if not os.path.exists(folder_dir):
                    os.makedirs(folder_dir)
                
            cv2.imwrite(os.path.join(folder_dir,img_file),draw_polygon)



# 의미정확성 FAIL 이미지 확인

import pandas as pd
import shutil

pd_data = pd.read_excel('/home/src/inspection/(036-049) 의미정확성 검사결과서_난소암 데이터_20230919.xlsx', sheet_name="TC1 검사이력", header=14, usecols = ["파일명", "검사 결과"])
data = pd_data.dropna(axis =0, how='any')
Fail_data = data[data["검사 결과"] == "FAIL"]["파일명"]

ori_path = "/home/data/draw_polygon/CT"
copy_path = "/home/data/draw_polygon/CT/error_image"

Fail_data = data[data["검사 결과"] == "FAIL"]["파일명"]

folder_list = os.listdir(ori_path)


for file_nm in data[data["검사 결과"] == "FAIL"]["파일명"]:
    
    ori_path_file = os.path.join(ori_path,f'{file_nm}.png')
    copy_path_file = os.path.join(copy_path,type,f'{file_nm}.png')
    
    # type = file_nm.split("_")[1]
    
    # if type == "CTA":

    #     copy_path_file = os.path.join(copy_path,type,f'{file_nm}.png')
        
    # elif type == "CTC":
        
    #     copy_path_file = os.path.join(copy_path,type,f'{file_nm}.png')

    shutil.copyfile(ori_path_file, copy_path_file)
        
