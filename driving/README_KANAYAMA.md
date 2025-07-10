# 자율주행 AI 칩 설계 - Driving 시스템 (Kanayama 제어기)

## 개요

이 시스템은 YOLOv3-Tiny 모델을 사용하여 차선을 탐지하고, Kanayama 제어기를 통해 정교한 차선 추종을 수행하는 자율주행 시스템입니다.

## 주요 기능

### 1. Kanayama 제어기
- 좌우 차선 정보를 모두 활용한 정교한 제어
- 횡방향 오차와 헤딩 오차를 동시에 고려한 제어
- 조향각 히스토리 관리로 안정적인 주행

### 2. 실시간 차선 탐지
- YOLO 모델을 통한 실시간 차선 탐지
- 좌우 차선 구분 및 기울기 계산
- 바운딩 박스 시각화

### 3. 모드 전환 기능
- 자율주행 모드 / 수동주행 모드

## 시스템 구조

```
driving/
├── main.py                      # 메인 실행 파일
├── driving_system_controller.py # 시스템 제어기
├── image_processor.py           # 이미지 처리 및 Kanayama 제어기
├── motor_controller.py          # 모터 제어
├── yolo_utils.py               # YOLO 모델 유틸리티
├── config.py                   # 설정 파일
└── AutoLab_lib.py              # 하드웨어 라이브러리
```

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install opencv-python numpy pynq pynq-dpu keyboard
```

### 2. 모델 파일 준비
- `../xmodel/top-tiny-yolov3_coco_256.xmodel`: YOLO 모델 파일
- `../xmodel/lane_class.txt`: 클래스 정의 파일

### 3. 실행
```bash
cd Autonomous-Driving-AI-Chip-Design/driving
python main.py
```

## 사용법

### 키보드 제어

| 키 | 기능 |
|---|---|
| `1` | 자율주행 모드 전환 |
| `2` | 수동주행 모드 전환 |
| `Space` | 주행 시작/정지 |
| `Q` | 프로그램 종료 |

### 수동 주행 모드에서 추가 제어

| 키 | 기능 |
|---|---|
| `W` | 전진 |
| `S` | 후진 |
| `A` | 좌회전 |
| `D` | 우회전 |
| `R` | 긴급 정지 |
