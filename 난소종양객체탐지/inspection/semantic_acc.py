
# 사전검증 의미정확성 확인

# 1. "암" 폴더 / "양성종양" 폴더에 각각 올바르게 라벨링 되어 있는지
# 2. "암" / "양성종양" 각각 올바른 종양 유형이 라벨링 되어 있는지
# 3. 이미지 1장 당 2개 이상의 라벨링이 되어있을 때, 종양 유형이 다른 경우가 있는지
# 4. 이미지 1장 당 2개 이상의 라벨링이 되어있을 때, 중복되어 기재된 경우가 있는지


import os
import json

first_dept = ["암","양성종양"]
second_dept = ["CT","초음파"]
ct_dept = ["골반","복부"]

root_dir = "/home/data/중간데이터/2.라벨링데이터"

cancer_type = []
benign_type = []


cancer_type_list = ['Clear cell carcinoma', 'Endometrioid carcinoma', 'Mucinous carcinoma', 'Etc', 'Serous carcinoma']
benign_type_list = ['Endometrioid tumor', 'Serous tumor', 'Mucinous tumor', 'Etc', 'Mature teratoma']

inspection_list = []

for first in first_dept:
    
    tmp_1_dir = os.path.join(root_dir,first)
    
    
    if first == "암":
        class_1 = "난소암"
    else:
        class_1 = "양성종양"
    
    for second in second_dept:
        
        target_dir = os.path.join(tmp_1_dir, second)
        
        if second == "CT":
            for ct in ct_dept:
                ct_target_dir = os.path.join(target_dir, ct)
                
                file_list = os.listdir(ct_target_dir)
                
                for file in sorted(file_list):
                    
                    label_dir = os.path.join(ct_target_dir, file)
                    
                    with open (label_dir, "r", encoding = "utf8") as f:
                        data = json.load(f)  
                        
                    taskname = data["taskName"]
                    resultData = data["resultData"]
                    
                    # 구분된 (암, 양성종양)폴더와 라벨링이 동일한가
                    check_num1 = (taskname == class_1)
                    
                    tumor_label_list = []
                    poly_list = []
                    for num, sub_data in enumerate(resultData):
                        
                        tumor_label = resultData[num]['value']
                        poly = resultData[num]['points']
                            
                        poly_list.append(poly)
                        
                        if first == "암":
                            # cancer_type.append(tumor_label
                            check_num2 = cancer_type_list.__contains__(tumor_label)
                            
                        else:
                            # benign_type.append(tumor_label)
                            check_num2 = benign_type_list.__contains__(tumor_label)
                            
                        tumor_label_list.append(tumor_label)
                            
                    if len(poly_list) > 1:
                        for iLoop in range(len(poly_list)-1):
                            
                            duplicate_poly = poly_list[iLoop] == poly_list[iLoop+1]
                            
                            if duplicate_poly == True:
                                check_num4 = False
                            else:
                                check_num4 = True
                    else:
                        check_num4 = True
                            
                    if len(set(tumor_label_list)) > 1:
                        check_num3 = False
                    else: check_num3 = True
                        
                    Pass_or_Fail_1 = "PASS" if check_num1 else "FAIL"
                    Pass_or_Fail_2 = "PASS" if check_num2 else "FAIL"
                    Pass_or_Fail_3 = "PASS" if check_num3 else "FAIL"
                    Pass_or_Fail_4 = "PASS" if check_num4 else "FAIL"
                
                    # 파일명, 1. 암/양성라벨링, 2.종양유형라벨링, 3.두개이상의 종양 유형, 4.중복된 폴리곤 라벨링
                    rst = [first, second, file, Pass_or_Fail_1, Pass_or_Fail_2, Pass_or_Fail_3, Pass_or_Fail_4]
                    inspection_list.append(rst)

        else:
            
            file_list = os.listdir(target_dir)
            
            for file in sorted(file_list):
                
                label_dir = os.path.join(target_dir, file)
                
                with open (label_dir, "r", encoding = "utf8") as f:
                    data = json.load(f)  
                    
                taskname = data["taskName"]
                resultData = data["resultData"]
                
                # 구분된 (암, 양성종양)폴더와 라벨링이 동일한가
                check_num1 = (taskname == class_1)
                
                tumor_label_list = []
                poly_list = []
                for num, sub_data in enumerate(resultData):
                    
                    tumor_label = resultData[num]['value']
                    poly = resultData[num]['points']
                        
                    poly_list.append(poly)
                    
                    if first == "암":
                        # cancer_type.append(tumor_label
                        check_num2 = cancer_type_list.__contains__(tumor_label)
                        
                    else:
                        # benign_type.append(tumor_label)
                        check_num2 = benign_type_list.__contains__(tumor_label)
                        
                    tumor_label_list.append(tumor_label)
                        
                if len(poly_list) > 1:
                    
                    for iLoop in range(len(poly_list)-1):
                        
                        duplicate_poly = poly_list[iLoop] == poly_list[iLoop+1]
                        
                        if duplicate_poly == True:
                            check_num4 = False
                        else:
                            check_num4 = True
                else:
                    check_num4 = True
                        
                        
                if len(set(tumor_label_list)) > 1:
                    check_num3 = False
                else: check_num3 = True
                    
                
                Pass_or_Fail_1 = "PASS" if check_num1 == True else "FAIL"
                Pass_or_Fail_2 = "PASS" if check_num2 == True else "FAIL"
                Pass_or_Fail_3 = "PASS" if check_num3 == True else "FAIL"
                Pass_or_Fail_4 = "PASS" if check_num4 == True else "FAIL"
                
                # 파일명, 1. 암/양성라벨링, 2.종양유형라벨링, 3.두개이상의 종양 유형, 4.중복된 폴리곤 라벨링
                rst = [first, second, file, Pass_or_Fail_1, Pass_or_Fail_2, Pass_or_Fail_3, Pass_or_Fail_4]
                inspection_list.append(rst)
    
    
    
import pandas as pd

df_rst = pd.DataFrame(inspection_list, columns=['task','data','file_nm', 'check_1', 'check_2','check_3','check_4'])

df_rst.to_csv("/home/src/inspection/images_inspection_mid.csv", index=False, encoding='cp949')



    
    
