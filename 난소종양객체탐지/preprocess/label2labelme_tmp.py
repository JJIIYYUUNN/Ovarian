import os
import json
import sys

from PIL import Image

class Label2Labelme():
  
  def __init__(self, data_type, label_input_path, png_input_path, save_path, CT_folder_nm = "골반", task = "cancer_benign"):
    
      self.data_type = data_type
      self.label_input_path = label_input_path
      self.png_input_path = png_input_path
      self.save_path = save_path
      self.CT_folder_nm = CT_folder_nm
      self.task = task
      
      self.cancer_object_cnt = 0
      self.benign_object_cnt = 0
      
      self.remake_format = { "version" : " 1.0.0",
          "flags": {},
          "shapes": [
            {
              "label": "example",
              "points": [
                [
                  0.0,
                  0.0
                ],
                [
                  0.0,
                  0.0
                ]
              ],
              "group_id": None,
              "shape_type": "polygon",
              "flags": {}
            }
          ],
          "imagePath": "example.png",
          "imageData": "example",
          "imageHeight": 0,
          "imageWidth": 0
        }
        
  # yolo 형식으로 바꾸기 위한 라벨링 형식 변경
  def change_label_structure(self):
    
    # cancer_object_cnt = 0
    # benign_object_cnt = 0
    
    for ca in ["암","양성종양"]:
      
      if self.data_type == "CT":
        
        root_path = f"{self.label_input_path}/{ca}/{self.data_type}/{self.CT_folder_nm}/"
        root_imagepath = f"{self.png_input_path}/{ca}/{self.data_type}/{self.CT_folder_nm}"
        
      elif self.data_type == "초음파":
        root_path = f"{self.label_input_path}/{ca}/{self.data_type}/"
        root_imagepath = f"{self.png_input_path}/{ca}/{self.data_type}"

      file_list = os.listdir(root_path)
      
      
      if ca == "암":
        cancer_file_cnt = len(file_list)
      else:
        benign_file_cnt = len(file_list)
      

      for file in sorted(file_list):
        label_path = os.path.join(root_path, file)

        with open (label_path, "r", encoding = "utf8") as f:
            data = json.load(f)

        filename = data["fileName"]
        taskname = data["taskName"]
        
        resultData = data["resultData"]
        
        if taskname == "난소암":
          eng_taskname = "cancer"
          self.cancer_object_cnt += len(resultData)
        elif taskname == "양성종양":
          eng_taskname = "benign"
          self.benign_object_cnt += len(resultData)
        else:
          print("Error")
          sys.exit()

        shapes_lst = []

        for num, detail in enumerate(resultData):
            
          label = resultData[num]['value']
          poly = resultData[num]['points']
          
          points_lst = []
          for pp in poly:
            x_point = pp["x"]
            y_point = pp["y"]
            points_lst.append([x_point, y_point])
          
          if self.task == "cancer_benign" :
            json_shapes = { "label" : eng_taskname, 
                          "points" : points_lst,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                          }
            
          elif self.task == "tumor_type":
            
            json_shapes = { "label" : label, 
                          "points" : points_lst,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                          }
          
          shapes_lst.append(json_shapes)
        
        
        imagepath = os.path.join(root_imagepath, f"{filename}.png")
        image1 = Image.open(imagepath)
        img_w, img_h = image1.size

        self.remake_format["shapes"] = shapes_lst
        self.remake_format["imagePath"] = imagepath
        self.remake_format["imageData"] = None
        self.remake_format["imageHeight"] = img_h
        self.remake_format["imageWidth"] = img_w

        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
        
        save_files = os.path.join(self.save_path, f"{filename}.json")
        with open( save_files, "w", encoding = "utf-8") as f:
            json.dump(self.remake_format, f, indent="\t", ensure_ascii=False)

    return cancer_file_cnt, self.cancer_object_cnt, benign_file_cnt, self.benign_object_cnt




  # 양성종양만 - yolo 형식으로 바꾸기 위한 라벨링 형식 변경
  def benign_change_label_structure(self):
    
    for ca in ["양성종양"]:
      
      if self.data_type == "CT":
        
        root_path = f"{self.label_input_path}/{ca}/{self.data_type}/{self.CT_folder_nm}/"
        root_imagepath = f"{self.png_input_path}/{ca}/{self.data_type}/{self.CT_folder_nm}"
        
      elif self.data_type == "초음파":
        root_path = f"{self.label_input_path}/{ca}/{self.data_type}/"
        root_imagepath = f"{self.png_input_path}/{ca}/{self.data_type}"

      file_list = os.listdir(root_path)
      
      
      if ca == "암":
        cancer_file_cnt = len(file_list)
        
      else:
        benign_file_cnt = len(file_list)
      

      for file in sorted(file_list):
        label_path = os.path.join(root_path, file)

        with open (label_path, "r", encoding = "utf8") as f:
            data = json.load(f)

        filename = data["fileName"]
        taskname = data["taskName"]
        
        resultData = data["resultData"]
        
        if taskname == "난소암":
          eng_taskname = "cancer"
          self.cancer_object_cnt += len(resultData)
        elif taskname == "양성종양":
          eng_taskname = "benign"
          self.benign_object_cnt += len(resultData)
        else:
          sys.exit()

        shapes_lst = []

        for num, detail in enumerate(resultData):
            
          label = resultData[num]['value']
          poly = resultData[num]['points']
          
          points_lst = []
          for pp in poly:
            x_point = pp["x"]
            y_point = pp["y"]
            points_lst.append([x_point, y_point])
          
          if self.task == "cancer_benign" :
            json_shapes = { "label" : eng_taskname, 
                          "points" : points_lst,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                          }
            
          elif self.task == "tumor_type":
            
            json_shapes = { "label" : label, 
                          "points" : points_lst,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                          }
          
          shapes_lst.append(json_shapes)
        
        
        imagepath = os.path.join(root_imagepath, f"{filename}.png")
        image1 = Image.open(imagepath)
        img_w, img_h = image1.size

        self.remake_format["shapes"] = shapes_lst
        self.remake_format["imagePath"] = imagepath
        self.remake_format["imageData"] = None
        self.remake_format["imageHeight"] = img_h
        self.remake_format["imageWidth"] = img_w

        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
        
        save_files = os.path.join(self.save_path, f"{filename}.json")
        with open( save_files, "w", encoding = "utf-8") as f:
            json.dump(self.remake_format, f, indent="\t", ensure_ascii=False)

    return benign_file_cnt, self.benign_object_cnt



class Label2Labelme_CT():
  
  def __init__(self, data_type, label_input_path, png_input_path, save_path, CT_folder_nm = "골반", task = "cancer_benign"):
    
      self.data_type = data_type
      self.label_input_path = label_input_path
      self.png_input_path = png_input_path
      self.save_path = save_path
      self.CT_folder_nm = CT_folder_nm
      self.task = task
      
      self.cancer_object_cnt = 0
      self.benign_object_cnt = 0
      
      self.remake_format = { "version" : " 1.0.0",
          "flags": {},
          "shapes": [
            {
              "label": "example",
              "points": [
                [
                  0.0,
                  0.0
                ],
                [
                  0.0,
                  0.0
                ]
              ],
              "group_id": None,
              "shape_type": "polygon",
              "flags": {}
            }
          ],
          "imagePath": "example.png",
          "imageData": "example",
          "imageHeight": 0,
          "imageWidth": 0
        }
        
  # yolo 형식으로 바꾸기 위한 라벨링 형식 변경
  def change_label_structure(self):
    
    # cancer_object_cnt = 0
    # benign_object_cnt = 0
    
    for ca in ["암","양성종양"]:
      
      for CT_folder_nm in ["골반", "복부"]:
        if self.data_type == "CT":
          
          root_path = f"{self.label_input_path}/{ca}/{self.data_type}/{CT_folder_nm}/"
          root_imagepath = f"{self.png_input_path}/{ca}/{self.data_type}/{CT_folder_nm}"
          
        elif self.data_type == "초음파":
          root_path = f"{self.label_input_path}/{ca}/{self.data_type}/"
          root_imagepath = f"{self.png_input_path}/{ca}/{self.data_type}"

        file_list = os.listdir(root_path)
        
        
        if ca == "암":
          cancer_file_cnt = len(file_list)
        else:
          benign_file_cnt = len(file_list)
        

        for file in sorted(file_list):
          label_path = os.path.join(root_path, file)

          with open (label_path, "r", encoding = "utf8") as f:
              data = json.load(f)

          filename = data["fileName"]
          taskname = data["taskName"]
          
          resultData = data["resultData"]
          
          if taskname == "난소암":
            eng_taskname = "cancer"
            self.cancer_object_cnt += len(resultData)
          elif taskname == "양성종양":
            eng_taskname = "benign"
            self.benign_object_cnt += len(resultData)
          else:
            print("Error")
            sys.exit()

          shapes_lst = []

          for num, detail in enumerate(resultData):
              
            label = resultData[num]['value']
            poly = resultData[num]['points']
            
            points_lst = []
            for pp in poly:
              x_point = pp["x"]
              y_point = pp["y"]
              points_lst.append([x_point, y_point])
            
            if self.task == "cancer_benign" :
              json_shapes = { "label" : eng_taskname, 
                            "points" : points_lst,
                              "group_id": None,
                              "shape_type": "polygon",
                              "flags": {}
                            }
              
            elif self.task == "tumor_type":
              
              json_shapes = { "label" : label, 
                            "points" : points_lst,
                              "group_id": None,
                              "shape_type": "polygon",
                              "flags": {}
                            }
            
            shapes_lst.append(json_shapes)
          
          
          imagepath = os.path.join(root_imagepath, f"{filename}.png")
          image1 = Image.open(imagepath)
          img_w, img_h = image1.size

          self.remake_format["shapes"] = shapes_lst
          self.remake_format["imagePath"] = imagepath
          self.remake_format["imageData"] = None
          self.remake_format["imageHeight"] = img_h
          self.remake_format["imageWidth"] = img_w

          if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
          
          save_files = os.path.join(self.save_path, f"{filename}.json")
          with open( save_files, "w", encoding = "utf-8") as f:
              json.dump(self.remake_format, f, indent="\t", ensure_ascii=False)

    return cancer_file_cnt, self.cancer_object_cnt, benign_file_cnt, self.benign_object_cnt





if __name__ == '__main__':
  
  # train, val, test for문 돌려야함 (train, val, test 폴더로 나눠졌을 때)
  
  data_type = "CT"
  label_input_path = "/home/data/first_data/2.라벨링데이터"
  png_input_path = "/home/data/first_data/1.원천데이터"
  save_path = "/home/data/tmp_data"
  
  if data_type == "CT":
    convertor = Label2Labelme(data_type, label_input_path, png_input_path, save_path)
    cancer_file_cnt, cancer_object_cnt, benign_file_cnt, benign_object_cnt = convertor.change_label_structure()
    
  else:
    convertor = Label2Labelme(data_type, label_input_path, png_input_path, save_path, task = "tumor_type")
    benign_file_cnt, benign_object_cnt = convertor.benign_change_label_structure()
  
  print("Success")
  # print(f" 난소암 - 파일 수 : {cancer_file_cnt} , 객체 수 : {cancer_object_cnt} \n 양성종양 - 파일 수 : {benign_file_cnt}, 객체 수 : {benign_object_cnt}")





