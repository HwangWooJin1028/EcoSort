import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk, messagebox, filedialog
import cv2
import torch
from PIL import Image, ImageTk
import numpy as np
import platform
import pathlib

class EcoSortApp:
    def __init__(self, root):
        self.root = root
        self.root.title("실시간 쓰레기 탐지 프로그램")
        root.configure(bg='white')

        # gpu cuda 사용할 시 gpu로, 아니면 cpu로 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 모델 로드 (pathlib 사용. 코드 도움: https://stackoverflow.com/questions/76994593/i-cant-load-my-yolov5-model-in-streamlit-this-is-the-error)
            if platform.system() == 'Windows':
                pathlib.PosixPath = pathlib.WindowsPath
            else:
                pathlib.WindowsPath = pathlib.PosixPath
            
            # 현재 파일의 디렉토리 절대 경로를 가져온다.
            current_dir = pathlib.Path(__file__).parent.resolve()
            # / 연산자를 이용해 경로를 쉽게 추가
            model_path = current_dir / "trash_model_best.pt"
            # Path 객체를 string으로
            model_path = str(model_path)
            # custom으로 model을 업로드
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            # 현재 모델의 클래스 이름을 영어에서 한글로 ({0: 'general waste', 1: 'hazardous waste', 2: 'organic', 3: 'recyclable'})
            self.model.names = {0: '일반쓰레기', 1: '유해쓰레기', 2: '음식물쓰레기', 3: '재활용쓰레기'}
            self.model.to(device=self.device) # cpu or gpu로 실행

        # 모델 로드 실패 시의 예외
        except Exception as e:
            messagebox.showerror("Error", f"모델 로드 실패: {str(e)}")
            root.destroy()
            return

        # 현재 컴퓨터에 연결되어 있는 카메라가 있는지 확인
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "웹캠을 열 수 없습니다.")
            root.destroy()
            return
        
        # root에 button, image, label 등을 추가.
        self.create_widgets()
        self.is_detecting = False
        self.image_path = None
    
    # program에 label 등을 추가
    def create_widgets(self):
        # 위아래 15px 간격
        # 버튼 3개를 탐는 컨테이너
        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=15) 
        
        # 비디오 탐지 버튼 추가
        self.toggle_button = ttk.Button(top_frame, text="비디오 탐지 시작", command=self.toggle_detection)
        self.toggle_button.pack(side=tk.LEFT, padx=5) # 좌우 5px 간격 및 왼쪽으로

        # 이미지 업로드 버튼 추가
        self.toggle_button_file = ttk.Button(top_frame, text="이미지 파일 업로드", command=self.image_file)
        self.toggle_button_file.pack(side=tk.LEFT, padx=5) # 좌우 5px 간격 및 왼쪽으로
        
        # 종료 버튼 추가
        quit_button = ttk.Button(top_frame, text="종료", command=self.quit_app)
        quit_button.pack(side=tk.LEFT, padx=5) # 좌우 5px 간격 및 왼쪽으로
        
        # 비디오 영상 이미지를 업로드하는 label 추가
        # top_frame 아래에 둠.
        self.video_frame = ttk.Label(self.root, background='white')
        self.video_frame.pack(padx=10, pady=15)
        
        # result label 추가
        # 비디오 프레임 아래에 둠.
        fontStyle = tkFont.Font(family="맑은 고딕", size=14)
        self.result_label = ttk.Label(self.root, text="탐지 결과가 여기에 표시됩니다", background='white', font=fontStyle)
        self.result_label.pack(pady=15)
        
    # 이미지 파일 버튼 연결 함수
    def image_file(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            # Pillow 라이브러리를 이용해 경로에 있는 이미지를 640 x 480 사이즈로 열기
            image = Image.open(self.image_path).resize((640, 480))
            # Tkinter에서 사용할 수 있는 형식(PhotoImage)으로 변환
            photo = ImageTk.PhotoImage(image=image)
            # 라벨 위젯에 표시될 이미지를 photo로 설정
            self.video_frame.config(image=photo)
            # 삭제될 위험(가비지컬렉션)을 지우기 위해 photo 객체를 라벨 위젯의 속성(self.video_frame.image)에 저장
            # Tkinter에서는 PhotoImage 객체가 참조되지 않는다면 메모리에서 삭제
            self.video_frame.image = photo
            self.result_label.config(text="이미지 파일이 업로드되었습니다. 탐지를 시작합니다")
            self.update_image()

    # 이미지 객체 탐지
    def update_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "먼저 이미지 파일을 업로드하세요.")
            return

        # 탐지를 하기 위해 이미지를 rgb 형식으로 변환
        image = Image.open(self.image_path).convert("RGB")
        # 탐지를 하기 위해 NumPy 배열 형태로
        image_np = np.array(image)
        
        ## 이미지 탐지 시작 ##
        results = self.model(image_np)
        annotated_frame = results.render()[0]
        detection_text = "탐지된 객체:\n"
        
        ## result_label 수정 ##
        # 'results.xyxy[0]'는 현재 프레임에서 탐지된 객체들의 리스트를 가지고 있다.
        for det in results.xyxy[0]: 
            # 각 탐지된 객체의 신뢰도 점수
            conf = float(det[4]) 
            # 탐지된 객체의 클래스 ID(객체의 종류)를 가져온다. 클래스 ID는 'det' 리스트의 여섯 번째 요소(인덱스 5)에 저장되어 있다.
            cls = int(det[5]) 
            # 객체 이름과 신뢰도를 텍스트 형식으로 표시하고, 신뢰도는 소수점 두 자리까지 표시한다.
            label = f"{self.model.names[cls]}: {conf:.2f}\n"  
            # 각 객체의 정보를 한 줄씩 'detection_text'에 추가하여 텍스트를 완성한다.
            detection_text += label  
        self.result_label.config(text=detection_text)

        # 이미지를 640 x 480으로 바꾸고 화면에 출력
        annotated_image = Image.fromarray(annotated_frame).resize((640, 480))
        photo = ImageTk.PhotoImage(image=annotated_image)
        self.video_frame.config(image=photo)
        self.video_frame.image = photo
     
    # 비디오 탐지 연결 함수
    def toggle_detection(self):
        # 만약 현재 탐지중이면 비디오 탐지 중지, 만약 탐지하고 있지 않다면 비디오 탐지 시작 후 객체탐지
        self.is_detecting = not self.is_detecting
        if self.is_detecting:
            self.toggle_button.config(text="비디오 탐지 중지")
            self.update_frame()
        else:
            self.toggle_button.config(text="비디오 탐지 시작")

    # 비디오 프레임 객체 탐지
    def update_frame(self):
        if self.is_detecting:
            # 현재 작동되고 있는 카메라의 한 프레임 받기
            ret, frame = self.cap.read()
            # 만약 정상적으로 잘 가지고 왔다면..
            if ret:
                
                ## 이미지 객체 탐지 및 바운딩 박스 그리기 ##

                # yolov5 모델에서 사용하기 위해 BGR -> RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(frame_rgb) # 객체 탐지 후 리턴
                
                # render() 메서드 사용하여 이미지에 레이블 및 바운딩 박스 그리기
                annotated_frame = results.render()[0]
                


                ## 탐지 결과 텍스트 업데이트##
                detection_text = "탐지된 객체:\n" 

                 # 현재 프레임에서 탐지된 객체 목록을 하나씩 반복한다. 'results.xyxy[0]'는 현재 프레임에서 탐지된 객체들의 리스트를 가지고 있다.
                for det in results.xyxy[0]: 
                     # 각 탐지된 객체의 신뢰도 점수를 가져온다
                    conf = float(det[4]) 
                     # 탐지된 객체의 클래스 ID(객체의 종류)를 가져온다. 클래스 ID는 'det' 리스트의 여섯 번째 요소(인덱스 5)에 저장되어 있다.
                    cls = int(det[5]) 
                    # 객체 이름과 신뢰도를 텍스트 형식으로 표시하고, 신뢰도는 소수점 두 자리까지 표시한다.
                    label = f"{self.model.names[cls]}: {conf:.2f}\n"  
                    # 각 객체의 정보를 한 줄씩 'detection_text'에 추가하여 텍스트를 완성한다.
                    detection_text += label  
                # UI에 있는 'result_label'을 업데이트하여 모든 탐지 결과 텍스트를 화면에 표시한다.
                self.result_label.config(text=detection_text)  


                ## 화면 업데이트 ##


                # 주석이 추가된 프레임(탐지 박스가 그려진 프레임)을 배열에서 PIL 이미지로 변환하고, 크기를 640x480으로 조정
                image = Image.fromarray(annotated_frame).resize((640, 480))  
                # 변환된 PIL 이미지를 Tkinter에서 사용 가능한 이미지 형식으로 변환한다
                photo = ImageTk.PhotoImage(image=image)  
                # UI의 'video_frame'을 업데이트하여 새 이미지가 표시되도록 한다.
                self.video_frame.config(image=photo)  
                # 이미지를 변수에 저장하여 가비지 컬렉션에 의해 제거되지 않도록 한다.
                self.video_frame.image = photo 

                # 일정 시간 간격으로 화면 갱신
                # 10밀리초 후에 'update_frame' 함수를 다시 호출하도록 예약
                # 이를 통해 프레임이 계속 갱신되며 실시간 화면을 구현
                self.root.after(10, self.update_frame)  


    # 카메라를 끈 후, 윈도우 화면 종료
    def quit_app(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EcoSortApp(root)
    root.mainloop()