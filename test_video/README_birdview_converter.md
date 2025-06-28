# 비디오를 버드아이 뷰 프레임으로 변환하기

이 코드는 자율주행 AI 칩 설계 프로젝트에서 사용하는 test_video를 위에서 바라본 것처럼 버드아이 뷰로 변환하여 frame 단위의 이미지로 만들어주는 도구입니다.

## 파일 설명

### 1. `simple_birdview_converter.py` (권장)
- 간단하고 사용하기 쉬운 버전
- 기본적인 버드아이 뷰 변환 기능 제공
- 특정 프레임 구간만 변환 가능

### 2. `video_to_birdview_frames.py`
- 더 상세한 기능을 제공하는 클래스 기반 버전
- 원본 프레임, ROI 이미지 등 추가 저장 옵션
- 더 많은 설정 옵션 제공

## 사용법

### 기본 사용법

```bash
cd Autonomous-Driving-AI-Chip-Design/test_video
python simple_birdview_converter.py
```

### 코드에서 직접 사용

```python
from simple_birdview_converter import convert_video_to_birdview_frames

# 전체 비디오 변환
convert_video_to_birdview_frames(
    video_path="test_video.mp4",
    output_dir="birdview_frames",
    output_size=(256, 256)
)

# 특정 구간만 변환
convert_video_to_birdview_frames(
    video_path="test_video.mp4",
    output_dir="birdview_frames_partial",
    output_size=(256, 256),
    start_frame=100,  # 100번째 프레임부터
    end_frame=200     # 200번째 프레임까지
)
```

## 출력 결과

### 생성되는 파일들
- `birdview_frames/` 디렉토리에 `frame_000001.jpg`, `frame_000002.jpg`, ... 형태로 저장
- 각 프레임은 256x256 크기의 버드아이 뷰 이미지
- YOLO 모델 입력에 바로 사용 가능한 형태

### 변환 과정
1. **원근 변환**: 카메라 시점을 위에서 바라보는 시점으로 변환
2. **ROI 추출**: 하단 부분만 추출 (차선 인식에 중요한 영역)
3. **크기 조정**: 256x256 크기로 리사이즈
4. **프레임 저장**: JPG 형태로 개별 저장

## 설정 옵션

### 원근 변환 좌표 조정
현재 코드의 좌표는 특정 카메라 설정에 맞춰져 있습니다. 다른 환경에서는 다음 좌표를 조정해야 할 수 있습니다:

```python
# src_mat: 원본 이미지에서의 4개 점 좌표
src_mat = np.float32([
    [238, 316],  # 좌상단
    [402, 313],  # 우상단
    [501, 476],  # 우하단
    [155, 476]   # 좌하단
])

# dst_mat: 변환 후 이미지에서의 4개 점 좌표
dst_mat = np.float32([
    [width * 0.3, 0],      # 좌상단
    [width * 0.7, 0],      # 우상단
    [width * 0.7, height], # 우하단
    [width * 0.3, height]  # 좌하단
])
```

### ROI 추출 위치 조정
```python
# 현재: 300픽셀부터 아래쪽만 추출
roi_image = bird_img[300:, :]

# 다른 위치로 변경하려면:
roi_image = bird_img[200:, :]  # 200픽셀부터
```

## 활용 예시

### 1. YOLO 모델 학습 데이터 생성
```python
# 버드아이 뷰 프레임들을 YOLO 학습 데이터로 활용
convert_video_to_birdview_frames(
    video_path="test_video.mp4",
    output_dir="yolo_training_data",
    output_size=(256, 256)
)
```

### 2. 차선 검출 테스트
```python
# 특정 구간만 변환하여 차선 검출 알고리즘 테스트
convert_video_to_birdview_frames(
    video_path="test_video.mp4",
    output_dir="lane_detection_test",
    output_size=(256, 256),
    start_frame=500,
    end_frame=600
)
```

### 3. 프레임들을 다시 비디오로 합치기
```python
from simple_birdview_converter import create_video_from_frames

create_video_from_frames(
    input_dir="birdview_frames",
    output_video_path="birdview_video.mp4",
    fps=30
)
```

## 주의사항

1. **메모리 사용량**: 긴 비디오의 경우 많은 디스크 공간이 필요할 수 있습니다.
2. **처리 시간**: 비디오 길이에 따라 변환 시간이 오래 걸릴 수 있습니다.
3. **좌표 조정**: 다른 카메라나 환경에서는 원근 변환 좌표를 조정해야 할 수 있습니다.
4. **OpenCV 설치**: `pip install opencv-python`이 필요합니다.

## 문제 해결

### 비디오를 열 수 없는 경우
- 비디오 파일 경로가 올바른지 확인
- 비디오 파일이 손상되지 않았는지 확인
- OpenCV가 제대로 설치되었는지 확인

### 변환된 이미지가 이상한 경우
- 원근 변환 좌표를 조정해보세요
- ROI 추출 위치를 조정해보세요
- 원본 비디오의 해상도를 확인해보세요

### 메모리 부족 오류
- 비디오를 더 작은 구간으로 나누어 처리해보세요
- 출력 이미지 크기를 줄여보세요 