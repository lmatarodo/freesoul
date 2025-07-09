#!/usr/bin/env python3
"""
Jupyter 환경에서 BEV 시각화를 테스트하기 위한 스크립트
PYNQ 기반 Jupyter 노트북에서 실행 가능
image_processor와 동일한 방식으로 직선 피팅 및 조향각 계산 포함
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
from collections import deque
from config import KANAYAMA_CONFIG, HISTORY_CONFIG

# Jupyter 환경 감지
def is_jupyter_environment():
    """Jupyter 환경인지 확인"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook
            return True
        elif shell == 'TerminalInteractiveShell':  # IPython terminal
            return False
        else:
            return False
    except NameError:
        return False

def slide_window_in_roi(binary, box, n_win=15, margin=30, minpix=10):
    """
    image_processor.py와 동일한 슬라이딩 윈도우 적용
    Args:
        binary: 2D np.array (전체 BEV 이진화 이미지)
        box: (y1, x1, y2, x2) – 전체 이미지 좌표계
        n_win: 윈도우 개수 (15)
        margin: 윈도우 마진 (30)
        minpix: 최소 픽셀 수 (10)
    Returns:
        fit: (slope, intercept), lane_pts: (x, y) 리스트
    """
    y1, x1, y2, x2 = box
    roi = binary[int(y1):int(y2), int(x1):int(x2)]
    if roi.size == 0:
        return None, None

    window_height = roi.shape[0] // n_win
    nonzero = roi.nonzero()
    nonzero_y, nonzero_x = nonzero
    left_inds = []

    # ROI 내부에서 히스토그램으로 초기 좌표 찾기
    hist = np.sum(roi[roi.shape[0]//2:,:], axis=0)
    if np.max(hist) > 0:
        current_x = np.argmax(hist)
    else:
        current_x = roi.shape[1] // 2

    for w in range(n_win):
        y_low = roi.shape[0] - (w+1)*window_height
        y_high = roi.shape[0] - w*window_height
        x_low = max(0, current_x-margin)
        x_high = min(roi.shape[1], current_x+margin)

        good_inds = ((nonzero_y>=y_low)&(nonzero_y<y_high)&(nonzero_x>=x_low)&(nonzero_x<x_high)).nonzero()[0]
        if len(good_inds) > minpix:
            current_x = int(nonzero_x[good_inds].mean())
            left_inds.append(good_inds)

    if left_inds:
        left_inds = np.concatenate(left_inds)
        leftx, lefty = nonzero_x[left_inds], nonzero_y[left_inds]
        # ROI 좌표를 전체 이미지 좌표로 변환
        leftx_global = leftx + int(x1)
        lefty_global = lefty + int(y1)
        if len(leftx_global) >= 2:
            left_fit = np.polyfit(lefty_global, leftx_global, 1)
            return left_fit, (leftx_global, lefty_global)
    return None, None

class LaneInfo:
    """차선 정보를 저장하는 클래스 (image_processor.py와 동일)"""
    def __init__(self, frame_width=256):
        self.left_x = frame_width // 2  # 왼쪽 차선 x좌표 (기본값: 차선 없음)
        self.right_x = frame_width // 2  # 오른쪽 차선 x좌표 (기본값: 차선 없음)
        self.left_slope = 0.0  # 왼쪽 차선 기울기
        self.left_intercept = 0.0  # 왼쪽 차선 y절편
        self.right_slope = 0.0  # 오른쪽 차선 기울기
        self.right_intercept = 0.0  # 오른쪽 차선 y절편
        self.left_points = None  # 슬라이딩 윈도우 결과 저장용
        self.right_points = None

def create_binary_image(gray_img):
    """image_processor.py와 동일한 이진화 이미지 생성"""
    # 그레이스케일을 BGR로 변환 (HSV 변환을 위해)
    if len(gray_img.shape) == 2:
        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    else:
        bgr_img = gray_img
    
    # 1) HSV로 흰색만 뽑기 (더 관대한 임계값)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 120])      # V 채널을 120으로 완화
    upper_white = np.array([180, 80, 255])   # S 채널을 80으로 완화
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    # 2) 그레이스케일 적응적 임계값
    mask_adaptive = cv2.adaptiveThreshold(gray_img, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    
    # 3) 고정 임계값도 백업으로 사용
    _, mask_gray = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)

    # 4) 세 마스크 결합
    mask = cv2.bitwise_or(mask_hsv, mask_adaptive)
    mask = cv2.bitwise_or(mask, mask_gray)

    # 5) 모폴로지로 노이즈 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask

def extract_lane_info_improved(boxes, processed_img):
    """image_processor.py와 동일한 차선 정보 추출"""
    h, w = processed_img.shape[:2]
    lane_info = LaneInfo(w)
    if len(boxes) == 0:
        return lane_info
    
    # 그레이스케일 변환
    if len(processed_img.shape) == 3:
        gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = processed_img
    
    left_lines = []
    right_lines = []
    
    for i, box in enumerate(boxes):
        y1, x1, y2, x2 = [int(v) for v in box]
        
        # 바운딩 박스 좌표가 이미지 범위를 벗어나는 경우 처리
        y1 = max(0, min(y1, processed_img.shape[0] - 1))
        x1 = max(0, min(x1, processed_img.shape[1] - 1))
        y2 = max(0, min(y2, processed_img.shape[0]))
        x2 = max(0, min(x2, processed_img.shape[1]))
        
        # 유효한 ROI 영역인지 확인
        if y1 >= y2 or x1 >= x2:
            continue
            
        roi = processed_img[y1:y2, x1:x2]
        
        # ROI가 비어있거나 유효하지 않은 경우 건너뛰기
        if roi is None or roi.size == 0:
            continue
            
        # ROI 크기가 너무 작은 경우도 건너뛰기
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            continue
            
        # image_processor.py와 동일한 처리
        blurred = cv2.GaussianBlur(roi, (5,5), 1)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        
        # 슬라이딩 윈도우 적용
        fit, pts = slide_window_in_roi(binary, (0, 0, binary.shape[0], binary.shape[1]), n_win=15, margin=30, minpix=10)
        
        if fit is not None and pts is not None:
            slope, intercept = fit
            xs, ys = pts
            
            # 이미지 하단(y=h-1)에서의 x 좌표 계산
            x_bottom = slope * (h - 1) + intercept
            line_info = {
                'x_bottom': x_bottom,
                'slope': slope,
                'intercept': intercept,
                'pixel_count': len(xs),
                'points': (xs, ys)
            }
        else:
            continue
        
        # 클래스 기반 좌우 분류 (가상 클래스 사용)
        image_center_x = w / 2
        if x_bottom < image_center_x:
            left_lines.append(line_info)
        else:
            right_lines.append(line_info)
    
    if left_lines:
        best_left = max(left_lines, key=lambda x: x['pixel_count'])
        lane_info.left_x = best_left['x_bottom']
        lane_info.left_slope = best_left['slope']
        lane_info.left_intercept = best_left['intercept']
        lane_info.left_points = best_left['points']
    
    if right_lines:
        best_right = max(right_lines, key=lambda x: x['pixel_count'])
        lane_info.right_x = best_right['x_bottom']
        lane_info.right_slope = best_right['slope']
        lane_info.right_intercept = best_right['intercept']
        lane_info.right_points = best_right['points']
    
    return lane_info

class TestVisualizationController:
    """image_processor.py와 동일한 제어 로직을 가진 테스트용 컨트롤러"""
    def __init__(self):
        # Kanayama 제어기 파라미터 (config에서 불러오기)
        self.K_y = KANAYAMA_CONFIG['K_y']
        self.K_phi = KANAYAMA_CONFIG['K_phi']
        self.L = KANAYAMA_CONFIG['L']
        self.lane_width = KANAYAMA_CONFIG['lane_width']
        self.v_r = KANAYAMA_CONFIG['v_r']
        
        # 조향각 히스토리 관리 (config에서 불러오기)
        self.steering_history = deque(maxlen=HISTORY_CONFIG['max_history_size'])
        self.no_lane_detection_count = 0
        self.max_no_lane_frames = HISTORY_CONFIG['max_no_lane_frames']
        self.default_steering_angle = HISTORY_CONFIG['default_steering_angle']
        self.avg_window_size = HISTORY_CONFIG['avg_window_size']
        self.smoothing_factor = HISTORY_CONFIG['smoothing_factor']
    
    def kanayama_control(self, lane_info):
        """image_processor.py와 동일한 Kanayama 제어기"""
        # 이미지 크기 (256x256으로 리사이즈됨)
        frame_width = 256
        
        # 1) 데이터 없으면 그대로
        if lane_info.left_x == frame_width // 2 and lane_info.right_x == frame_width // 2:
            print("차선을 찾을 수 없습니다.")
            return 0.0, KANAYAMA_CONFIG['v_r']  # config에서 v_r 사용
        
        lane_width_m = 0.9  # image_processor.py와 동일한 차로 폭
        Fix_Speed = KANAYAMA_CONFIG['v_r']  # config에서 v_r 사용
        
        # 2) 픽셀 단위 차로 폭 & 픽셀당 미터 변환 계수
        lane_pixel_width = lane_info.right_x - lane_info.left_x
        if lane_pixel_width > 0:
            pix2m = lane_pixel_width / lane_width_m
        else:
            pix2m = frame_width / lane_width_m  # fallback

        # 3) 횡방향 오차: 차량 중앙(pixel) - 차로 중앙(pixel) → m 단위
        image_cx = frame_width / 2.0
        lane_cx = (lane_info.left_x + lane_info.right_x) / 2.0
        lateral_err = (lane_cx - image_cx) / pix2m

        # 4) 헤딩 오차 (차선 기울기 평균)
        heading_err = -0.5 * (lane_info.left_slope + lane_info.right_slope)

        # 5) Kanayama 제어식 (image_processor.py와 동일한 파라미터)
        K_y, K_phi, L = 0.1, 0.3, 0.5
        v_r = Fix_Speed
        v = v_r * (math.cos(heading_err))**2
        w = v_r * (K_y * lateral_err + K_phi * math.sin(heading_err))
        delta = math.atan2(w * L, v)

        # 6) 픽셀→도 단위 보정 (k_p) - image_processor.py와 동일
        steering = math.degrees(delta) * (Fix_Speed/25)
        steering = max(min(steering, 30.0), -30.0)
        
        # 디버깅 정보 출력
        print(f"Lateral error: {lateral_err:.3f}m, Heading error: {math.degrees(heading_err):.1f}°")
        print(f"Lane center: {lane_cx:.1f}, Image center: {image_cx:.1f}")
        print(f"Steering: {steering:.2f}°, Speed: {v:.1f}")
        print()  # 빈 줄 추가
        
        return steering, v
    
    def add_steering_to_history(self, steering_angle):
        """조향각을 히스토리에 추가"""
        self.steering_history.append(steering_angle)
        
    def get_average_steering(self, num_frames=None):
        """최근 N개 조향각의 평균 계산"""
        if len(self.steering_history) == 0:
            return self.default_steering_angle
        
        # 기본값 사용
        if num_frames is None:
            num_frames = self.avg_window_size
        
        # 최근 N개 값만 사용
        recent_values = list(self.steering_history)[-min(num_frames, len(self.steering_history)):]
        
        # 평균 계산
        average_steering = sum(recent_values) / len(recent_values)
        
        return average_steering
        
    def get_smoothed_steering(self, current_steering_angle):
        """스무딩을 적용한 조향각 계산"""
        if len(self.steering_history) == 0:
            return current_steering_angle
        
        # 이전 값과 현재 값을 가중 평균
        previous_angle = self.steering_history[-1]
        smoothed_angle = (self.smoothing_factor * previous_angle + 
                         (1 - self.smoothing_factor) * current_steering_angle)
        
        return smoothed_angle
        
    def should_use_history(self, lane_info):
        """히스토리 사용 여부 결정"""
        # 이미지 크기 (256x256으로 리사이즈됨)
        frame_width = 256
        
        # 양쪽 차선이 모두 보이지 않는 경우
        if lane_info.left_x == frame_width // 2 and lane_info.right_x == frame_width // 2:
            self.no_lane_detection_count += 1
            return True
        else:
            # 차선이 보이면 카운터 리셋
            self.no_lane_detection_count = 0
            return False
    
    def get_robust_steering_angle(self, lane_info, base_steering_angle):
        """강건한 조향각 계산 (히스토리 적용)"""
        # 히스토리 사용 여부 결정
        if self.should_use_history(lane_info):
            # 히스토리에서 평균값 사용
            if len(self.steering_history) > 0:
                robust_angle = self.get_average_steering()
                print(f"히스토리 평균 조향각 사용: {robust_angle:.2f}°")
            else:
                robust_angle = self.default_steering_angle
                print(f"기본 조향각 사용: {robust_angle:.2f}°")
        else:
            # 현재 계산된 조향각 사용
            robust_angle = base_steering_angle
            
            # 스무딩 적용
            if len(self.steering_history) > 0:
                robust_angle = self.get_smoothed_steering(robust_angle)
        
        # 히스토리에 추가
        self.add_steering_to_history(robust_angle)
        
        return robust_angle

def draw_boxes_on_bev(bev_img, bev_boxes, color=(0, 255, 0)):
    """BEV 영상 위에 바운딩 박스를 그리는 함수"""
    if bev_img is None:
        return None
    
    img = bev_img.copy()
    for box in bev_boxes:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        class_name = box['class']
        score = box['score']
        
        # 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 클래스명과 점수 표시
        label = f"{class_name}: {score:.2f}"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def test_single_frame_visualization():
    """단일 프레임으로 BEV 시각화 테스트 (직선 피팅 + 조향각 계산 포함)"""
    print("단일 프레임 BEV 시각화 테스트를 시작합니다...")
    print("=" * 60)
    
    # 테스트용 컨트롤러 초기화
    controller = TestVisualizationController()
    
    # 비디오 파일 열기
    video_path = '../result.mp4'
    if not os.path.exists(video_path):
        print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        return
    
    # BEV 변환 파라미터 (image_processor.py와 동일)
    h, w = frame.shape[0], frame.shape[1]
    dst_mat = [[round(w * 0.3), 0], [round(w * 0.7), 0], 
              [round(w * 0.7), h], [round(w * 0.3), h]]
    src_mat = [[250, 316], [380, 316], [450, 476], [200, 476]]
    
    print(f"원본 프레임 크기: {frame.shape}")
    print(f"BEV 변환 파라미터:")
    print(f"  src_mat: {src_mat}")
    print(f"  dst_mat: {dst_mat}")
    print("-" * 60)
    
    # BEV 변환
    transform_matrix = cv2.getPerspectiveTransform(
        np.float32(src_mat), 
        np.float32(dst_mat)
    )
    bird_img = cv2.warpPerspective(frame, transform_matrix, (w, h))
    roi_image = bird_img[300:, :]  # ROI 영역
    
    print(f"BEV 영상 크기: {bird_img.shape}")
    print(f"ROI 영역 크기: {roi_image.shape}")
    print("-" * 60)
    
    # 256x256 리사이즈 (YOLO 모델 입력용)
    img_256 = cv2.resize(roi_image, (256, 256))
    
    # 가상의 바운딩 박스 생성 (테스트용) - 256x256 리사이즈된 좌표
    # 실제로는 YOLO 모델이 256x256 이미지에서 예측한 좌표
    boxes_256x256 = [
        {
            'x1': 50, 'y1': 100, 'x2': 150, 'y2': 200,  # 256x256 좌표
            'class': 'car', 'score': 0.85
        },
        {
            'x1': 180, 'y1': 120, 'x2': 220, 'y2': 180,  # 256x256 좌표
            'class': 'lane', 'score': 0.92
        }
    ]
    
    print("가상 바운딩 박스 (256x256 좌표):")
    for i, box in enumerate(boxes_256x256):
        print(f"  Box {i+1}: ({box['x1']}, {box['y1']}) - ({box['x2']}, {box['y2']}) - {box['class']} ({box['score']:.2f})")
    print("-" * 60)
    
    # 256x256 좌표를 원본 BEV ROI 크기로 스케일 조정
    scale_x = roi_image.shape[1] / 256.0
    scale_y = roi_image.shape[0] / 256.0
    
    bev_boxes = []
    for box in boxes_256x256:
        bev_boxes.append({
            'x1': int(box['x1'] * scale_x),
            'y1': int(box['y1'] * scale_y),
            'x2': int(box['x2'] * scale_x),
            'y2': int(box['y2'] * scale_y),
            'class': box['class'],
            'score': box['score']
        })
    
    print("스케일 조정된 바운딩 박스 (BEV ROI 좌표):")
    for i, box in enumerate(bev_boxes):
        print(f"  Box {i+1}: ({box['x1']}, {box['y1']}) - ({box['x2']}, {box['y2']}) - {box['class']} ({box['score']:.2f})")
    print("-" * 60)
    
    # === 직선 피팅 및 차선 정보 추출 ===
    print("직선 피팅 및 차선 정보 추출:")
    print("-" * 60)
    
    # 바운딩 박스를 numpy 배열로 변환 (image_processor 형식)
    boxes_np = []
    for box in boxes_256x256:
        boxes_np.append([box['y1'], box['x1'], box['y2'], box['x2']])  # (y1, x1, y2, x2) 형식
    
    # 차선 정보 추출
    lane_info = extract_lane_info_improved(boxes_np, img_256)
    
    print(f"왼쪽 차선:")
    print(f"  X 좌표: {lane_info.left_x:.1f}")
    print(f"  기울기: {lane_info.left_slope:.3f}")
    print(f"  Y절편: {lane_info.left_intercept:.1f}")
    print(f"  픽셀 수: {len(lane_info.left_points[0]) if lane_info.left_points else 0}")
    
    print(f"오른쪽 차선:")
    print(f"  X 좌표: {lane_info.right_x:.1f}")
    print(f"  기울기: {lane_info.right_slope:.3f}")
    print(f"  Y절편: {lane_info.right_intercept:.1f}")
    print(f"  픽셀 수: {len(lane_info.right_points[0]) if lane_info.right_points else 0}")
    print("-" * 60)
    
    # === Kanayama 제어기로 조향각 계산 ===
    print("Kanayama 제어기 조향각 계산:")
    print("-" * 60)
    
    # 기본 조향각과 속도 계산
    base_steering_angle, calculated_speed = controller.kanayama_control(lane_info)
    
    # 강건한 조향각 계산 (히스토리 적용)
    steering_angle = controller.get_robust_steering_angle(lane_info, base_steering_angle)
    
    print(f"기본 조향각: {base_steering_angle:.2f}°")
    print(f"최종 조향각: {steering_angle:.2f}°")
    print(f"계산된 속도: {calculated_speed:.1f} m/s")
    print(f"히스토리 크기: {len(controller.steering_history)}")
    print("-" * 60)
    
    # === 시각화 ===
    print("시각화 생성 중...")
    
    if is_jupyter_environment():
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BEV Visualization with Lane Detection & Steering', fontsize=16)
        
        # 1. 원본 프레임
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        # 2. BEV 변환 영상
        axes[0, 1].imshow(cv2.cvtColor(bird_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Bird\'s Eye View (Full)')
        axes[0, 1].axis('off')
        
        # 3. ROI 영역
        axes[0, 2].imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('ROI Region')
        axes[0, 2].axis('off')
        
        # 4. 256x256 리사이즈 (YOLO 입력)
        axes[1, 0].imshow(cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('256x256 Resized (YOLO Input)')
        axes[1, 0].axis('off')
        
        # 5. BEV + 바운딩 박스
        bev_with_boxes = draw_boxes_on_bev(roi_image, bev_boxes)
        axes[1, 1].imshow(cv2.cvtColor(bev_with_boxes, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'BEV with Bounding Boxes\nDetected: {len(bev_boxes)} objects')
        axes[1, 1].axis('off')
        
        # 6. 시스템 정보
        info_text = []
        info_text.append(f"Base Steering: {base_steering_angle:.2f}°")
        info_text.append(f"Final Steering: {steering_angle:.2f}°")
        info_text.append(f"Speed: {calculated_speed:.1f} m/s")
        info_text.append(f"Left Lane X: {lane_info.left_x:.1f}")
        info_text.append(f"Right Lane X: {lane_info.right_x:.1f}")
        info_text.append(f"Left Slope: {lane_info.left_slope:.3f}")
        info_text.append(f"Right Slope: {lane_info.right_slope:.3f}")
        info_text.append(f"History Size: {len(controller.steering_history)}")
        info_text.append(f"Detected Objects: {len(bev_boxes)}")
        
        axes[1, 2].text(0.1, 0.9, '\n'.join(info_text), transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('System Information')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    else:
        # 일반 환경에서 OpenCV 사용
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Bird\'s Eye View', bird_img)
        cv2.imshow('ROI Region', roi_image)
        cv2.imshow('256x256 Resized', img_256)
        
        bev_with_boxes = draw_boxes_on_bev(roi_image, bev_boxes)
        cv2.imshow('BEV with Bounding Boxes', bev_with_boxes)
        
        # 정보 텍스트
        info_img = np.zeros((350, 500, 3), dtype=np.uint8)
        cv2.putText(info_img, f"Base Steering: {base_steering_angle:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_img, f"Final Steering: {steering_angle:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_img, f"Speed: {calculated_speed:.1f} m/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(info_img, f"Left X: {lane_info.left_x:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(info_img, f"Right X: {lane_info.right_x:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(info_img, f"History Size: {len(controller.steering_history)}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(info_img, f"Objects: {len(bev_boxes)}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        cv2.imshow('System Info', info_img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("시각화 완료!")
    print("=" * 60)

def test_video_visualization(max_frames=30):
    """비디오로 BEV 시각화 테스트 (직선 피팅 + 조향각 계산 포함)"""
    print(f"비디오 BEV 시각화 테스트를 시작합니다... (최대 {max_frames} 프레임)")
    print("=" * 60)
    
    # 테스트용 컨트롤러 초기화
    controller = TestVisualizationController()
    
    # 비디오 파일 열기
    video_path = '../result.mp4'
    if not os.path.exists(video_path):
        print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    # BEV 변환 파라미터
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dst_mat = [[round(w * 0.3), 0], [round(w * 0.7), 0], 
              [round(w * 0.7), h], [round(w * 0.3), h]]
    src_mat = [[250, 316], [380, 316], [450, 476], [200, 476]]
    
    transform_matrix = cv2.getPerspectiveTransform(
        np.float32(src_mat), 
        np.float32(dst_mat)
    )
    
    frame_count = 0
    
    if is_jupyter_environment():
        plt.figure(figsize=(15, 10))
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BEV 변환
        bird_img = cv2.warpPerspective(frame, transform_matrix, (w, h))
        roi_image = bird_img[300:, :]
        img_256 = cv2.resize(roi_image, (256, 256))
        
        # 가상의 바운딩 박스 생성 (테스트용) - 256x256 리사이즈된 좌표
        bev_boxes = []
        if frame_count % 10 == 0:  # 10프레임마다 박스 생성
            # 256x256 좌표에서 예측된 박스 (실제 YOLO 모델 출력)
            box_256x256 = {
                'x1': 50 + frame_count, 'y1': 100, 'x2': 150 + frame_count, 'y2': 200,
                'class': 'car', 'score': 0.85
            }
            
            # 256x256 좌표를 원본 BEV ROI 크기로 스케일 조정
            scale_x = roi_image.shape[1] / 256.0
            scale_y = roi_image.shape[0] / 256.0
            
            bev_boxes.append({
                'x1': int(box_256x256['x1'] * scale_x),
                'y1': int(box_256x256['y1'] * scale_y),
                'x2': int(box_256x256['x2'] * scale_x),
                'y2': int(box_256x256['y2'] * scale_y),
                'class': box_256x256['class'],
                'score': box_256x256['score']
            })
        
        # 차선 정보 추출 및 조향각 계산
        boxes_np = []
        for box in bev_boxes:
            # BEV ROI 좌표를 256x256 좌표로 역변환
            scale_x = 256.0 / roi_image.shape[1]
            scale_y = 256.0 / roi_image.shape[0]
            boxes_np.append([
                int(box['y1'] * scale_y), 
                int(box['x1'] * scale_x), 
                int(box['y2'] * scale_y), 
                int(box['x2'] * scale_x)
            ])
        
        lane_info = extract_lane_info_improved(boxes_np, img_256)
        
        # 기본 조향각과 속도 계산
        base_steering_angle, calculated_speed = controller.kanayama_control(lane_info)
        
        # 강건한 조향각 계산 (히스토리 적용)
        steering_angle = controller.get_robust_steering_angle(lane_info, base_steering_angle)
        
        # 진행 상황 출력
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: Base {base_steering_angle:.1f}°, Final {steering_angle:.1f}°, Speed {calculated_speed:.1f} m/s")
        
        if is_jupyter_environment():
            plt.clf()
            
            # 2x3 서브플롯
            plt.subplot(2, 3, 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f'Original Frame {frame_count}')
            plt.axis('off')
            
            plt.subplot(2, 3, 2)
            plt.imshow(cv2.cvtColor(bird_img, cv2.COLOR_BGR2RGB))
            plt.title('Bird\'s Eye View')
            plt.axis('off')
            
            plt.subplot(2, 3, 3)
            plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
            plt.title('ROI Region')
            plt.axis('off')
            
            plt.subplot(2, 3, 4)
            plt.imshow(cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB))
            plt.title('256x256 Resized')
            plt.axis('off')
            
            plt.subplot(2, 3, 5)
            if len(bev_boxes) > 0:
                bev_with_boxes = draw_boxes_on_bev(roi_image, bev_boxes)
                plt.imshow(cv2.cvtColor(bev_with_boxes, cv2.COLOR_BGR2RGB))
                plt.title(f'BEV with Boxes\nObjects: {len(bev_boxes)}')
            else:
                plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
                plt.title('BEV with Boxes\nNo objects')
            plt.axis('off')
            
            plt.subplot(2, 3, 6)
            info_text = [
                f"Base Steering: {base_steering_angle:.1f}°",
                f"Final Steering: {steering_angle:.1f}°",
                f"Speed: {calculated_speed:.1f} m/s",
                f"Left X: {lane_info.left_x:.1f}",
                f"Right X: {lane_info.right_x:.1f}",
                f"History: {len(controller.steering_history)}",
                f"Objects: {len(bev_boxes)}"
            ]
            plt.text(0.1, 0.9, '\n'.join(info_text), transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            plt.title('System Info')
            plt.axis('off')
            
            plt.tight_layout()
            plt.pause(0.1)
        
        frame_count += 1
    
    cap.release()
    
    if is_jupyter_environment():
        plt.close()
    
    print(f"비디오 처리 완료! 총 {frame_count} 프레임 처리됨")
    print("=" * 60)

if __name__ == '__main__':
    print("BEV 시각화 테스트 스크립트 (직선 피팅 + 조향각 계산 포함)")
    print("=" * 60)
    
    # Jupyter 환경 확인
    if is_jupyter_environment():
        print("✅ Jupyter 환경에서 실행 중입니다.")
        print("matplotlib을 사용하여 이미지를 표시합니다.")
    else:
        print("⚠️ 일반 Python 환경에서 실행 중입니다.")
        print("OpenCV를 사용하여 이미지를 표시합니다.")
    
    print("\n사용 가능한 테스트:")
    print("1. test_single_frame_visualization() - 단일 프레임 BEV 시각화 테스트")
    print("2. test_video_visualization(max_frames=30) - 비디오 BEV 시각화 테스트")
    
    print("\n예시:")
    print("test_single_frame_visualization()  # 단일 프레임 테스트")
    print("test_video_visualization(20)  # 20프레임 비디오 테스트") 