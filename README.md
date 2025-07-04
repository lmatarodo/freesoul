# Autonomous Driving AI Chip Design Baseline File Structure

2025 Autonomous Driving AI Chip design contest for Sungkyunkwan University Students

---

## Folder Structure

### $\textcolor{red}{\mathbf{(NEW)}}$ Segmentation

- **test_data**  
  *Inference test img*
- **dpu_yolact.ipynb**  
  *Jupyter Notebook for real-time segmentation Inference (~20FPS)*

### debugging
- **SoC_Driving.ipynb**  
  *Jupyter Notebook for driving code*
- **data_collection.ipynb**  
  *Jupyter Notebook for data collection*
- $\textcolor{red}{\mathbf{(NEW)}}$ **test_sonic.ipynb**  
  *Jupyter Notebook for Ultrasonic sensor test (for reference only as it's a simple example)*

### dpu
- **dpu.bit**  
  *DPU bitstream file **(Students need to add)***
- **dpu.hwh**  
  *DPU hardware file **(Students need to add)***
- **dpu.xclbin**  
  *DPU executable file **(Students need to add)***

### driving
- **config.py**  
  *Initial motor address settings*
- **driving_system_controller.py**  
  *Driving mode settings*
- **image_processor.py**  
  *Image processing script*
- **main.py**  
  *Driving parameter settings*
- **motor_controller.py**  
  *Motor control settings*
- **yolo_utils.py**  
  *YOLO utility functions*

### test_video
- **test_video.mp4**  
  *Test video file*

### xmodel
- **lane_class.txt**  
  *Model class configuration **(Students need to add)***
- **top-tiny-yolov3_coco_256.xmodel**  
  *Compiled deep learning model file **(Students need to add)***

---


# AutoLab SoC Driving - Student Modification Guide

This section outlines the key parts of each file that students need to modify. Review the details and file locations carefully before making changes.

---

## 1. main.py

Here are the lines of code you need to update:

```python
# Line 36
speed = 0

# Line 37
steering_speed = 50

# Line 47
overlay = DpuOverlay("./dpu/dpu.bit")

# Line 48
overlay.load_model("./xmodel/top-tiny-yolov3_coco_256.xmodel")
```

---

## 2. config.py

Modify the following parts of the file:

```python
# Line 9
MOTOR_ADDRESSES = {
    'motor_0': 0x00A0000000,
    'motor_1': 0x00A0010000,
    'motor_2': 0x00A0020000,
    'motor_3': 0x00A0030000,
    'motor_4': 0x00A0040000,
    'motor_5': 0x00A0050000
}

# Line 18
ADDRESS_RANGE = 0x10000

# Line 23
classes_path = "./xmodel/lane_class.txt"
```

---

## 3. motor_controller.py

You need to update lines 53-54 and make changes to specific functions:

```python
# Lines 53-54
register_most_left, register_most_right
```

### Functions to modify:
- **def right()**  
- **def left()**  
- **def stay()**  
- **def set_left_speed()**  
- **def set_right_speed()**

### Motor Control Algorithm:
- **def control_motors()**

---

## 4. image_processor.py

Update the following variables in the file:

```python
# Variable settings
self.reference_point_x = 190
self.reference_point_y = 150
self.point_detection_height = 20
```

---

## Notes

- Double-check file names and line numbers before making changes.  
- Save all changes and test the project to confirm everything works correctly.

# 자율주행 AI 칩 설계 프로젝트

## 프로젝트 개요

이 프로젝트는 YOLOv3 기반 차선 검출과 Kanayama 제어기를 사용한 자율주행 시스템입니다. 카메라 영상에서 차선을 인식하고, 조향각을 계산하여 모터를 제어하는 방식으로 동작합니다.

## 주요 구성 요소

### 1. 핵심 모듈

- **`driving_system_controller.py`**: 메인 주행 시스템 컨트롤러
- **`image_processor.py`**: YOLO 추론 및 차선 검출 처리
- **`motor_controller.py`**: 모터 제어 로직
- **`yolo_utils.py`**: YOLOv3 후처리 유틸리티
- **`visualize_lane_prediction.py`**: 차선 검출 시각화 및 테스트

### 2. 차선 검출 알고리즘

#### 기존 방식
- HSV 색공간과 그레이스케일 조합으로 흰색 차선 픽셀 추출
- `cv2.fitLine` 또는 `np.polyfit`을 사용한 직선 피팅
- 모폴로지 연산으로 노이즈 제거

#### 개선된 슬라이딩 윈도우 알고리즘 (신규)
```python
def slide_window_in_roi(binary, box, n_win=9, margin=20, minpix=20):
    """
    슬라이딩 윈도우를 사용한 차선 검출
    
    Args:
        binary: 2D np.array (전체 BEV 이진화 이미지)
        box: (x1, y1, x2, y2) – 전체 이미지 좌표계
        n_win: 윈도우 개수
        margin: 윈도우 마진
        minpix: 최소 픽셀 수
        
    Returns:
        fit: (slope, intercept), lane_pts: (x, y) 리스트
    """
```

**슬라이딩 윈도우 알고리즘 특징:**
1. **히스토그램 기반 초기 위치**: ROI 하단부의 히스토그램으로 차선의 초기 x 위치 추정
2. **단계적 윈도우 검색**: 하단부터 상단까지 여러 개의 윈도우로 차선을 추적
3. **적응적 중심점 업데이트**: 각 윈도우에서 검출된 픽셀의 평균 위치로 다음 윈도우 중심 조정
4. **강건한 직선 피팅**: 수집된 모든 픽셀에 대해 `np.polyfit`으로 직선 피팅
5. **이상치 제거**: 최소 픽셀 수 조건으로 노이즈 필터링

**장점:**
- 곡선 차선에도 강건한 검출
- 노이즈와 조명 변화에 강함
- 차선의 연속성 보장
- 기존 방식과의 호환성 유지

**좌표계 통일 (최신 개선사항):**
- 모든 박스 좌표를 `(x1, y1, x2, y2)` 형식으로 통일
- 슬라이딩 윈도우와 직선 피팅에서 일관된 좌표계 사용
- 직선을 박스 내부로 클리핑하여 시각화 개선

### 3. 제어 시스템

#### Kanayama 제어기
```python
# 제어 파라미터
self.v_r = 0.5      # 기준 속도 (m/s)
self.L = 0.2        # 휠베이스 (m)
self.K_y = 0.5      # 횡방향 오차 게인
self.K_phi = 1.0    # 방향각 오차 게인
self.lane_width = 0.3  # 차로 폭 (m)
```

**제어 로직:**
1. **차로 중앙 주행**: 양쪽 차선이 모두 있을 때
2. **단일 차선 추종**: 한쪽 차선만 있을 때 반대쪽 차로 경계 추정
3. **속도 분모 안정화**: 큰 조향각에서 속도 감소
4. **히스토리 기반 스무딩**: 차선 검출 실패 시 이전 조향각 사용

### 4. 데이터 수집

#### 수동 주행 모드
- 키보드 조작으로 수동 주행
- 이미지, 속도, 조향각 정보 자동 저장
- YOLO 학습을 위한 데이터셋 구축

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install opencv-python numpy matplotlib
```

### 2. YOLO 모델 설정
- 학습된 YOLO 모델 파일을 `xmodel/` 디렉토리에 배치
- 클래스 파일을 `xmodel/lane_class.txt`에 설정

### 3. 실행 방법

#### 주행 시스템 실행
```bash
cd driving
python driving_system_controller.py
```

#### 시각화 테스트
```bash
cd driving
python visualize_lane_prediction.py
```

#### 슬라이딩 윈도우 테스트
```bash
cd driving
python test_sliding_window.py <이미지/비디오 경로>
```

## 테스트 및 검증

### 1. 시각화 도구
- **`visualize_lane_prediction.py`**: 실시간 차선 검출 시각화
- **`test_sliding_window.py`**: 슬라이딩 윈도우 알고리즘 테스트
- 조향각, 속도, 차선 정보 실시간 표시

### 2. 성능 지표
- 차선 검출 정확도
- 조향각 안정성
- 주행 안정성 (히스토리 기반)

## 주요 개선 사항

### 1. 차선 검출 강화
- HSV + 그레이스케일 조합 필터링
- 모폴로지 연산으로 노이즈 제거
- 슬라이딩 윈도우 알고리즘 도입
- **좌표계 통일**: 모든 박스 좌표를 `(x1, y1, x2, y2)` 형식으로 통일
- **임계값 최적화**: V 채널 150→120, S 채널 60→80으로 완화
- **적응적 임계값**: `cv2.adaptiveThreshold` 적용으로 조명 변화에 강화
- **다중 마스크 결합**: HSV + 적응적 임계값 + 고정 임계값 조합

### 2. 제어기 안정화
- 속도 분모 안정화
- 히스토리 기반 스무딩
- 긴급 회피 모드

### 3. 시각화 개선
- 실시간 차선 검출 결과 표시
- 슬라이딩 윈도우 포인트 시각화
- 제어 모드 및 파라미터 표시
- **직선 클리핑**: 차선 직선을 박스 내부로 제한하여 시각화 개선
- **슬라이딩 윈도우 박스 표시**: 각 윈도우를 개별적으로 시각화
- **이진화 결과 표시**: 흰색 픽셀 비율과 개수 실시간 표시
- **별도 이진화 창**: 'b' 키로 이진화 결과 토글 가능

## 파일 구조

```
Autonomous-Driving-AI-Chip-Design/
├── driving/
│   ├── driving_system_controller.py  # 메인 주행 시스템
│   ├── image_processor.py            # 이미지 처리 및 차선 검출
│   ├── motor_controller.py           # 모터 제어
│   ├── visualize_lane_prediction.py  # 시각화 도구
│   └── test_sliding_window.py        # 슬라이딩 윈도우 테스트
├── xmodel/
│   ├── lane_class.txt                # YOLO 클래스 정의
│   └── [YOLO 모델 파일들]
├── test_video/                       # 테스트 비디오
└── README.md
```

## 주의사항

1. **YOLO 모델**: 현재 학습 전 상태이며, 학생들이 직접 학습된 모델을 추가해야 합니다.
2. **차선 라벨링**: 오른쪽 차선만 탐지하므로 데이터 수집 시 오른쪽 차선만 라벨링하면 됩니다.
3. **하드웨어 설정**: 모터 제어를 위한 하드웨어 연결이 필요합니다.
4. **조명 조건**: 조명 변화에 강하도록 설계되었지만, 극단적인 조명 조건에서는 성능이 저하될 수 있습니다.

## 향후 개선 방향

1. **다중 차선 검출**: 좌우 차선 모두 검출하여 더 정확한 주행
2. **곡선 차선 대응**: 고차 다항식 피팅으로 곡선 차선 검출 강화
3. **딥러닝 기반 검출**: CNN 기반 차선 검출 모델 도입
4. **센서 융합**: 라이다, IMU 등 추가 센서 정보 활용

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.
