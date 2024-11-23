# EcoSort
Yolov5 기반 실시간 쓰레기 분류 프로그램, EcoSort
<br/><br/><br/>


## 개요
- Yolov5 모델 기반으로 웹캠을 이용하여 실시간으로 쓰레기를 분류한다
- 또한, 사용자가 업로드한 사진 파일을 이용하여 쓰레기를 분류한다.
- 분류한 쓰레기는 label을 통해 보여준다
<br/><br/><br/>

## 개발 환경
![image](https://github.com/user-attachments/assets/9fca1bed-c185-4c44-9fb2-120b50175313)

<br/><br/><br/>


## 모델 학습
[<img src="https://img.shields.io/badge/Colab-F7DF1E.svg?style=for-the-badge&logo=googlecolab&logoColor=#F9AB00"/>](https://github.com/HwangWooJin1028/EcoSort/blob/main/EcoSort_%EC%93%B0%EB%A0%88%EA%B8%B0%ED%83%90%EC%A7%80.ipynb)
[<img src="https://img.shields.io/badge/roboflow-5C2D91?style=for-the-badge&logo=roboflow&logoColor=white">](https://universe.roboflow.com/ecotrack/ecotrack)
- colab을 통해 모델 학습
- roboflow를 통해 데이터 획득
- 자세한 모델 학습 코드는 위의 COLAB 사진 클릭
<br/><br/><br/>


## Native App
- colab을 통해 학습된 모델을 로드하여 객체 탐지
- 실시간 웹캠 프레임 및 이미지를 객체탐지하여 화면에 출력
- 자세한 코드는 EcoSort.py 코드 확인
<br/><br/><br/>

## UI 화면
&nbsp;&nbsp;📌 화면 UI <br/>
![image](https://github.com/user-attachments/assets/c0173888-d68a-4b8b-bd32-4793d40ef538)
<br/><br/><br/>

## 결과 화면
&nbsp;&nbsp;📌 실시간 웹캠 감지 <br/>
![image](https://github.com/user-attachments/assets/73565adb-f0a3-4060-b2f4-2610279b7fea)
<br/><br/><br/>

&nbsp;&nbsp;📌 업로드 사진 감지 <br/>
![image](https://github.com/user-attachments/assets/8ff7e0cd-42c5-4b21-9086-f25a6530207a)

<br/><br/><br/>

