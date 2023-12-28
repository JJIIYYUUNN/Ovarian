
<img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white"><img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
<br>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"><img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">

### 난소암 데이터 구축 사업(2023) 

#### 🔔사업 개요
 > **난소종양을 가진 환자에서 악성 여부 예측 및 치료 성적 예측 서비스 모델 개발에 활용할 수 있는 난소암 및 난소 양성종양 의료영상 구축
(난소종양 초음파 및 CT영상, Whole Slide Image(WSI) 데이터, 암표지자검사 데이터, 메타데이터)**
- 데이터 명 : 난소암 데이터(Data-Building Project in Ovary)
- 데이터 분야 : 헬스케어
- 데이터 유형 : 이미지
- 데이터 소개 : 난소 종양을 가진 환자의 난소 종양부 CT, 초음파 이미지 및 난소암 Whole Slide Image(WSI)와 암표지자검사결과 및 의료기록정보들로부터 획득한 난소종양 환자의 종합 의료 데이터


<br>

#### 🔔주요 내용

 > **CT와 초음파 이미지를 활용하여 종양 영역(폴리곤)을 탐지하고 클래스를 예측하는 모델 개발 수행**
- 사용 알고리즘 : YOLOv8
- 학습 이미지 예시
  |CT|초음파|
  |:--:|:--:|
  |![image](https://github.com/JJIIYYUUNN/Ovarian/assets/125724830/0c38ffe0-e013-4177-b4b7-f7ccc88026b1)|![image](https://github.com/JJIIYYUUNN/Ovarian/assets/125724830/b3e5c758-4b29-45cd-a774-924a4db4d18d)|

<br>

- 모델 성능
  |CT 종양 영역 탐지 및 악성여부 예측|초음파 양성종양 영역 탐지 및 종양 유형(5개) 예측|
  |:--:|:--:|
  |95.5%(map50)|80.3% (map50)|
  |![image](https://github.com/JJIIYYUUNN/Ovarian/assets/125724830/dcfc1918-d45f-43af-8492-4851ed6e5247)|![image](https://github.com/JJIIYYUUNN/Ovarain/assets/125724830/a38c85bd-3b82-4bf8-8e3c-67b33d503b41)|

<br>

- 데이터 수량
 1. CT 종양 영역 탐지 및 악성여부 예측 모델
    + 파일수
      |Class|Train|Val|Test|Total|
      |:--:|:--:|:--:|:--:|:--:|
      |암|8,734|1,075|1,134|10,943|
      |양성종양|30,252|3,799|3,740|37,791|
    + 객체수
      |Class|Train|Val|Test|Total|
      |:--:|:--:|:--:|:--:|:--:|
      |암|10,409|1,285|1,335|13,029|
      |양성종양|30,595|3,843|3,781|38,219|

<br>
      
 2. 초음파 양성종양 영역 탐지 및 종양 유형 예측 모델
    + 파일수
      |Class|Train|Val|Test|Total|
      |:--:|:--:|:--:|:--:|:--:|
      |자궁내막종|2,366|299|299|2,964|
      |장액성|1,150|163|137|1,450|
      |점액성|1,700|206|223|2,129|
      |성숙기형종|2,928|349|378|3,655|
      |기타|2,064|259|239|2,562|
      
    + 객체수
      |Class|Train|Val|Test|Total|
      |:--:|:--:|:--:|:--:|:--:|
      |자궁내막종|2,377|301|300|2,978|
      |장액성|1,207|172|142|1,521|
      |점액성|1,831|219|233|2,283|
      |성숙기형종|3,191|374|410|3,975|
      |기타|2,109|263|245|2,617|

<br>

 - 모델 학습 및 평가 명령어 예시
 1. CT 종양 영역 탐지 및 악성여부 예측 모델
      + 데이터 전처리
        ```
        python /home/src/preprocess/l2y.py --json_dir /home/data/tmp_data --val_size 0.1 --test_size 0.1 --label_list cancer benign --data_type CT
        ```
      + 모델 학습
        ```
        python /home/src/yolov8/yolo8_ready_seg.py --task train
        ```
      + 모델 평가
        ```
        python /home/src/yolov8/yolo8_ready_seg.py --task test --model_path [학습모델위치]
        ```

 2. 초음파 양성종양 영역 탐지 및 종양 유형 예측 모델
      + 데이터 전처리
        ```
        python /home/src/preprocess/l2y.py --json_dir /home/data/tmp_data --val_size 0.1 --test_size 0.1 --label_list 'Mature teratoma' 'Mucinous tumor' 'Serous tumor' 'Endometrioid tumor' 'Etc' --data_type 초음파
        ```
      + 모델 학습
        ```
        python /home/src/yolov8/yolo8_ready_seg.py --task train
        ```
      + 모델 평가
        ```
        python /home/src/yolov8/yolo8_ready_seg.py --task test --model_path [학습모델위치]
        ```

    
