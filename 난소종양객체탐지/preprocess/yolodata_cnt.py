import os
import json
import pandas as pd
from tabulate import tabulate

import yaml


with open('/home/data/tmp_data/YOLODataset/dataset.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

if len(_cfg['names']) == 2:

    # 양/악성 데이터 수량 파악
    path = "/home/data/tmp_data/YOLODataset/labels"

    total_file = 0
    total_object = 0

    df = pd.DataFrame([],columns=["Train", "Val", "Test"], index=[["암","암","양성종양","양성종양","합계","합계"], ["파일수","객체수","파일수","객체수","파일수","객체수"]])

    for enu, sp in enumerate(['train','val','test']):
        
        cancer_file_cnt = 0
        benign_file_cnt = 0
        
        cancer_object_cnt = 0
        benign_object_cnt = 0
        
        total_file_cnt = 0
        total_object_cnt = 0
        
        target_path = os.path.join(path,sp)
        
        file_list = os.listdir(target_path)
        
        
        for file in file_list:
            
            total_file += 1
            
            file_path = os.path.join(target_path,file)
            
            with open (file_path, "r", encoding = "utf8") as f:
                data = f.readlines()
            
            total_object += len(data)
            
            # 파일수
            total_file_cnt += 1
            if data[0][0] == '0':
                cancer_file_cnt += 1
            else:
                benign_file_cnt += 1
            
            # 객체수
            total_object_cnt += len(data)
            
            for line in range(len(data)):
                
                if data[line][0] == '0':
                    cancer_object_cnt += 1
                else:
                    benign_object_cnt += 1
        
        df.iloc[0][enu] = cancer_file_cnt
        df.iloc[2][enu] = benign_file_cnt
        df.iloc[4][enu] = total_file_cnt
        
        df.iloc[1][enu] = cancer_object_cnt
        df.iloc[3][enu] = benign_object_cnt
        df.iloc[5][enu] = total_object_cnt
        
        # print(f"(%s) - %5s >>> %5s,%5s " % ("파일", sp, cancer_file_cnt, benign_file_cnt))
        # print(f"(%s) - %5s >>> %5s,%5s \n" % ("객체", sp, cancer_object_cnt, benign_object_cnt))

    print(tabulate(df))

    print(f"(%s) - %5s" % ("전체 파일", total_file))
    print(f"(%s) - %5s" % ("전체 객체", total_object))


else:
# 종양유형 5개 클래스 데이터 수량 파악

# with open('/home/data/tmp_data/YOLODataset/dataset.yaml', encoding='UTF-8') as f:
#     _cfg = yaml.load(f, Loader=yaml.FullLoader)

    class_nm = _cfg['names']

    class_nm.append("Etc")
    class_nm.append("Etc_2")

    path = "/home/data/tmp_data/YOLODataset/labels"

    total_file = 0
    total_object = 0

    df = pd.DataFrame([],columns=["Train", "Val", "Test"], index=[[class_nm[0],class_nm[0],class_nm[1],class_nm[1],class_nm[2],class_nm[2],class_nm[3],class_nm[3],class_nm[4],class_nm[4],"합계","합계"], 
                                                                ["파일수","객체수","파일수","객체수","파일수","객체수","파일수","객체수","파일수","객체수","파일수","객체수"]])

    for enu, sp in enumerate(['train','val','test']):
        
        file_cnt_0, file_cnt_1, file_cnt_2, file_cnt_3, file_cnt_4 = 0, 0, 0, 0, 0
        
        total_file_cnt, total_object_cnt = 0, 0
        
        object_cnt_0, object_cnt_1, object_cnt_2, object_cnt_3, object_cnt_4 = 0, 0, 0, 0, 0
        
        target_path = os.path.join(path,sp)
        
        file_list = os.listdir(target_path)
        
        for file in file_list:
            
            total_file += 1
            
            file_path = os.path.join(target_path,file)
            
            with open (file_path, "r", encoding = "utf8") as f:
                data = f.readlines()
            
            total_object += len(data)
            
            # 파일수
            total_file_cnt += 1
            if data[0][0] == '0':
                file_cnt_0 += 1
                
            elif data[0][0] == "1":
                file_cnt_1 += 1
                
            elif data[0][0] == "2":
                file_cnt_2 += 1

            elif data[0][0] == "3":
                file_cnt_3 += 1
                
            else:
                file_cnt_4 += 1
            
            # 객체수
            total_object_cnt += len(data)
            
            for line in range(len(data)):
                
                if data[line][0] == '0':
                    object_cnt_0 += 1
                    
                elif data[line][0] == '1':
                    object_cnt_1 += 1
                    
                elif data[line][0] == '2':
                    object_cnt_2 += 1

                elif data[line][0] == '3':
                    object_cnt_3 += 1
                    
                else:
                    object_cnt_4 += 1
        
        df.iloc[0][enu] = file_cnt_0
        df.iloc[2][enu] = file_cnt_1
        df.iloc[4][enu] = file_cnt_2
        df.iloc[6][enu] = file_cnt_3
        df.iloc[8][enu] = file_cnt_4
        df.iloc[10][enu] = total_file_cnt
        
        df.iloc[1][enu] = object_cnt_0
        df.iloc[3][enu] = object_cnt_1
        df.iloc[5][enu] = object_cnt_2
        df.iloc[7][enu] = object_cnt_3
        df.iloc[9][enu] = object_cnt_4
        df.iloc[11][enu] = total_object_cnt
        
        # print(f"(%s) - %5s >>> %5s,%5s " % ("파일", sp, cancer_file_cnt, benign_file_cnt))
        # print(f"(%s) - %5s >>> %5s,%5s \n" % ("객체", sp, cancer_object_cnt, benign_object_cnt))

    print(tabulate(df))

    print(f"(%s) - %5s" % ("전체 파일", total_file))
    print(f"(%s) - %5s" % ("전체 객체", total_object))