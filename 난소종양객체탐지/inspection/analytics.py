
# 사전검증 다양성 확인

import os


# 원천데이터-라벨링데이터 매칭 확인
root_path = "/home/data/사전검증"

png_lst = []
label_lst = []

for png_or_label in ["1.원천데이터","2.라벨링데이터"]:
    
    one_path = os.path.join(root_path, png_or_label)
    
    for ca in ["암","양성종양"]:
        two_path = os.path.join(one_path, ca)
        
        for target in ["CT","초음파"]:
            three_path = os.path.join(two_path, target)
            
            if target == "CT":
                
                for last in ["골반","복부"]:
                    last_path = os.path.join(three_path, last)
                    
                    file_list = os.listdir(last_path)
                    
                    if png_or_label == "1.원천데이터":
                        png_lst.append(file_list)
                    else:
                        label_lst.append(file_list)
            
            else:
                file_list = os.listdir(three_path)
                
                if png_or_label == "1.원천데이터":
                    png_lst.append(file_list)
                else:
                    label_lst.append(file_list)
                

        
png_lst[0]
png_lst[1]
png_lst[2]
png_lst[3]