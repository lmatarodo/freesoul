# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import cv2
import numpy as np
import math
import os
import colorsys
import random
from PIL import Image
import time
from collections import deque
from yolo_utils import pre_process, evaluate
from config import KANAYAMA_CONFIG, HISTORY_CONFIG

def slide_window_in_roi(binary, box, n_win=15, margin=30, minpix=10):
    """
    debugging/visualize.py의 slide_window_search_roi와 동일하게 슬라이딩 윈도우 적용
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
    """차선 정보를 저장하는 클래스"""
    def __init__(self):
        self.left_x = 130  # 왼쪽 차선 x좌표 (기본값: 차선 없음)
        self.right_x = 130  # 오른쪽 차선 x좌표 (기본값: 차선 없음)
        self.left_slope = 0.0  # 왼쪽 차선 기울기
        self.left_intercept = 0.0  # 왼쪽 차선 y절편
        self.right_slope = 0.0  # 오른쪽 차선 기울기
        self.right_intercept = 0.0  # 오른쪽 차선 y절편
        # fitLine 파라미터 추가
        self.left_params = None   # (vx, vy, x0, y0)
        self.right_params = None  # (vx, vy, x0, y0)
        self.left_points = None  # 슬라이딩 윈도우 결과 저장용
        self.right_points = None

class ImageProcessor:
    def __init__(self, dpu, classes_path, anchors):
        # 클래스 변수로 저장
        self.dpu = dpu
            
        self.classes_path = classes_path
        self.anchors = anchors
        self.class_names = self.load_classes(classes_path)    
        
        self.reference_point_x = 200
        self.reference_point_y = 240
        self.point_detection_height = 20
        
        # DPU 초기화 상태 추적 플래그
        self.initialized = False
        self.init_dpu()
        
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
        
    def load_classes(self, classes_path):
        """Load class names from the given path"""
        with open(classes_path, 'r') as f:
            class_names = f.read().strip().split('\n')
        return class_names 
    
    def init_dpu(self):
        """DPU 초기화 - 한 번만 실행됨"""
        if self.initialized:
            print("DPU 이미 초기화됨")
            return  # 이미 초기화되었으면 바로 리턴

        print("DPU 초기화 중...")
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()

        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut0 = tuple(outputTensors[0].dims)
        self.shapeOut1 = tuple(outputTensors[1].dims)

        outputSize0 = int(outputTensors[0].get_data_size() / self.shapeIn[0])
        outputSize1 = int(outputTensors[1].get_data_size() / self.shapeIn[0])

        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]
        self.output_data = [
            np.empty(self.shapeOut0, dtype=np.float32, order="C"),
            np.empty(self.shapeOut1, dtype=np.float32, order="C")
        ]

        # 초기화 완료 플래그 설정
        self.initialized = True
        print("DPU 초기화 완료")

    
    def roi_rectangle_below(self, img, cutting_idx):
        return img[cutting_idx:, :]

    def warpping(self, image, srcmat, dstmat):
        h, w = image.shape[0], image.shape[1]
        transform_matrix = cv2.getPerspectiveTransform(srcmat, dstmat)
        minv = cv2.getPerspectiveTransform(dstmat, srcmat)
        _image = cv2.warpPerspective(image, transform_matrix, (w, h))
        return _image, minv
    
    def bird_convert(self, img, srcmat, dstmat):
        srcmat = np.float32(srcmat)
        dstmat = np.float32(dstmat)
        img_warpped, _ = self.warpping(img, srcmat, dstmat)
        return img_warpped

    def calculate_angle(self, x1, y1, x2, y2):
        if x1 == x2:
            return 90.0
        slope = (y2 - y1) / (x2 - x1)
        return -math.degrees(math.atan(slope))

    def color_filter(self, image):
        """참고 코드와 동일한 흰색 픽셀 검출 방식"""
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 흰색 픽셀 검출 (참고 코드와 동일한 방식)
        lower = np.array([0, 255, 255])  # 흰색 필터
        upper = np.array([255, 255, 255])
        white_mask = cv2.inRange(hsv, lower, upper)
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        
        return masked

    def create_binary_image(self, gray_img):
        """HSV + 그레이스케일 조합으로 이진화 이미지 생성 (개선된 임계값)"""
        # 그레이스케일을 BGR로 변환 (HSV 변환을 위해)
        if len(gray_img.shape) == 2:
            bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        else:
            bgr_img = gray_img
        
        # 1) HSV로 흰색만 뽑기 (더 관대한 임계값)
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])      # V 채널을 150 → 120으로 완화
        upper_white = np.array([180, 80, 255])   # S 채널을 60 → 80으로 완화
        mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

        # 2) 그레이스케일 적응적 임계값 (더 강건한 방법)
        # 기존 고정 임계값 대신 적응적 임계값 사용
        mask_adaptive = cv2.adaptiveThreshold(gray_img, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        
        # 추가로 고정 임계값도 백업으로 사용 (더 낮은 임계값)
        _, mask_gray = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)  # 150 → 120으로 완화

        # 3) 세 마스크 결합 (더 강건한 검출)
        mask = cv2.bitwise_or(mask_hsv, mask_adaptive)
        mask = cv2.bitwise_or(mask, mask_gray)

        # 4) 모폴로지로 노이즈 정리 (더 부드러운 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # 작은 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # 작은 노이즈 제거
        
        return mask

    def extract_lane_pixels_improved(self, gray_img, box):
        """개선된 차선 픽셀 추출 (개선된 임계값)"""
        y1, x1, y2, x2 = box
        
        # 바운딩 박스 영역 추출
        roi = gray_img[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return []
        
        # BGR 변환 (HSV를 위해)
        if len(roi.shape) == 2:
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        else:
            roi_bgr = roi
        
        # 1) 적응적 HSV 필터링 (더 관대한 범위)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])      # V 채널을 120으로 완화
        upper_white = np.array([180, 80, 255])   # S 채널을 80으로 완화
        mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

        # 2) 그레이스케일 적응적 임계값
        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mask_adaptive = cv2.adaptiveThreshold(gray_roi, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)

        # 3) 세 마스크 결합
        mask = cv2.bitwise_or(mask_hsv, mask_adaptive)
        
        # 추가로 고정 임계값도 백업으로 사용
        _, mask_gray = cv2.threshold(gray_roi, 120, 255, cv2.THRESH_BINARY)  # 150 → 120으로 완화
        mask = cv2.bitwise_or(mask, mask_gray)

        # 4) 부드러운 모폴로지 연산 (더 작은 커널)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 5) 픽셀 좌표 추출
        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) < 5:  # 최소 픽셀 수를 5로 낮춤
            return []
        
        pts = pts.reshape(-1, 2)  # Nx2, (x, y) 순서
        
        # ROI 좌표계를 전체 이미지 좌표계로 변환
        pts[:, 0] += int(x1)  # x 좌표
        pts[:, 1] += int(y1)  # y 좌표
        
        return pts

    def extract_lane_info_improved(self, xyxy_results, classes_results, processed_img):
        """debugging/visualize.py의 process_roi와 동일하게 ROI 내 이진화, 슬라이딩 윈도우, 직선 피팅 적용"""
        lane_info = LaneInfo()
        if len(xyxy_results) == 0:
            return lane_info
        # 그레이스케일 변환
        if len(processed_img.shape) == 3:
            gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = processed_img
        h, w = gray_img.shape[:2]
        left_lines = []
        right_lines = []
        for i, box in enumerate(xyxy_results):
            y1, x1, y2, x2 = [int(v) for v in box]
            roi = processed_img[y1:y2, x1:x2]
            # === debugging/visualize.py의 process_roi와 동일하게 ===
            blurred = cv2.GaussianBlur(roi, (5,5), 1)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
            # ROI 하단 20% 마스킹
            h_roi = binary.shape[0]
            binary[h_roi*80//100:, :] = 0
            # 전체 프레임 좌표에 합성은 생략 (필요시 추가)
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
                    'vx': 0, 'vy': 0, 'x0': 0, 'y0': 0,
                    'pixel_count': len(xs),
                    'points': (xs, ys)
                }
            else:
                continue
            # 클래스 기반 좌우 분류
            class_id = classes_results[i] if i < len(classes_results) else 0
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else ""
            if "left" in class_name.lower():
                left_lines.append(line_info)
            elif "right" in class_name.lower():
                right_lines.append(line_info)
            else:
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
            lane_info.left_params = (best_left['vx'], best_left['vy'], best_left['x0'], best_left['y0'])
            lane_info.left_points = best_left['points']
        if right_lines:
            best_right = max(right_lines, key=lambda x: x['pixel_count'])
            lane_info.right_x = best_right['x_bottom']
            lane_info.right_slope = best_right['slope']
            lane_info.right_intercept = best_right['intercept']
            lane_info.right_params = (best_right['vx'], best_right['vy'], best_right['x0'], best_right['y0'])
            lane_info.right_points = best_right['points']
        return lane_info

    
    def kanayama_control(self, lane_info):
        """debugging/visualize.py와 동일한 Kanayama 제어기"""
        # 1) 데이터 없으면 그대로
        if lane_info.left_x == 130 and lane_info.right_x == 130:
            print("차선을 찾을 수 없습니다.")
            # 히스토리에서 평균값 사용
            if len(self.steering_history) > 0:
                avg_steering = self.get_average_steering()
                print(f"히스토리 평균 조향각 사용: {avg_steering:.2f}°")
                return avg_steering, self.v_r
            else:
                return self.default_steering_angle, self.v_r
        
        # 이미지 크기 (256x256으로 리사이즈됨)
        frame_width = 256
        lane_width_m = 3.5  # debugging/visualize.py와 동일한 차로 폭
        Fix_Speed = self.v_r
        
        # 2) 픽셀 단위 차로 폭 & 픽셀당 미터 변환 계수
        lane_pixel_width = lane_info.right_x - lane_info.left_x
        if lane_pixel_width > 0:
            pix2m = lane_pixel_width / lane_width_m
        else:
            pix2m = frame_width / lane_width_m  # fallback

        # 3) 횡방향 오차: 차량 중앙(pixel) - 차로 중앙(pixel) → m 단위
        image_cx = frame_width / 2.0
        lane_cx = (lane_info.left_x + lane_info.right_x) / 2.0
        lateral_err = (image_cx - lane_cx) / pix2m

        # 4) 헤딩 오차 (차선 기울기 평균)
        heading_err = -0.5 * (lane_info.left_slope + lane_info.right_slope)

        # 5) Kanayama 제어식 (debugging/visualize.py와 동일한 파라미터)
        K_y, K_phi, L = 0.3, 0.9, 0.5
        v_r = Fix_Speed
        v = v_r * (math.cos(heading_err))**2
        w = v_r * (K_y * lateral_err + K_phi * math.sin(heading_err))
        delta = math.atan2(w * L, v)

        # 6) 픽셀→도 단위 보정 (k_p) - debugging/visualize.py와 동일
        steering = math.degrees(delta) * (Fix_Speed/25)
        steering = max(min(steering, 50.0), -50.0)
        
        # 디버깅 정보 출력
        print(f"Lateral error: {lateral_err:.3f}m, Heading error: {math.degrees(heading_err):.1f}°")
        print(f"Lane center: {lane_cx:.1f}, Image center: {image_cx:.1f}")
        print(f"Steering: {steering:.2f}°, Speed: {v:.1f}")
        
        return steering, v

    def detect_lane_center_x(self, xyxy_results):
        """기존 방식: 가장 오른쪽 차선의 중심점 반환 (하위 호환성 유지)"""
        rightmost_lane_x_min = None
        rightmost_lane_x_max = None
        rightmost_x_position = -float('inf')
        
        for box in xyxy_results:
            y1, x1, y2, x2 = box
            if x1 > rightmost_x_position:
                rightmost_x_position = x1
                rightmost_lane_x_min = int(x1)
                rightmost_lane_x_max = int(x2)
        
        if rightmost_lane_x_min is not None and rightmost_lane_x_max is not None:
            return (rightmost_lane_x_min + rightmost_lane_x_max) // 2
        return None

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
        # 양쪽 차선이 모두 보이지 않는 경우
        if lane_info.left_x == 130 and lane_info.right_x == 130:
            self.no_lane_detection_count += 1
            return True
        else:
            # 차선이 보이면 카운터 리셋
            self.no_lane_detection_count = 0
            return False
            
    def get_robust_steering_angle(self, lane_info, current_steering_angle):
        """강건한 조향각 계산 (히스토리 + 스무딩 적용)"""
        if self.should_use_history(lane_info):
            if self.no_lane_detection_count <= self.max_no_lane_frames:
                # 이전 값들의 평균 사용
                robust_angle = self.get_average_steering()
                print(f"차선 미검출: 이전 {min(self.avg_window_size, len(self.steering_history))}개 값 평균 사용 ({robust_angle:.2f}°)")
                return robust_angle
            else:
                # 너무 오래 차선을 못 찾으면 기본값 사용
                print(f"차선 미검출: 기본값 사용 ({self.default_steering_angle:.2f}°)")
                return self.default_steering_angle
        else:
            # 차선이 보이면 스무딩 적용
            smoothed_angle = self.get_smoothed_steering(current_steering_angle)
            
            # 히스토리에 추가 (스무딩된 값)
            self.add_steering_to_history(smoothed_angle)
            
            # 스무딩이 적용되었는지 출력
            if len(self.steering_history) > 1:
                print(f"스무딩 적용: {current_steering_angle:.2f}° → {smoothed_angle:.2f}°")
            
            return smoothed_angle

    def process_frame(self, img, use_kanayama=True):
        """프레임 처리 및 조향각 계산"""
        h, w = img.shape[0], img.shape[1]
        dst_mat = [[round(w * 0.3), 0], [round(w * 0.7), 0], 
                  [round(w * 0.7), h], [round(w * 0.3), h]]
        src_mat = [[238, 316], [402, 313], [501, 476], [155, 476]]
        
        bird_img = self.bird_convert(img, srcmat=src_mat, dstmat=dst_mat)
        roi_image = self.roi_rectangle_below(bird_img, cutting_idx=300)
        
        img = cv2.resize(roi_image, (256, 256))
        image_size = img.shape[:2]
        image_data = np.array(pre_process(img, (256, 256)), dtype=np.float32)
        
        start_time = time.time()

        # self를 사용하여 클래스 변수에 접근
        self.input_data[0][...] = image_data.reshape(self.shapeIn[1:])
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        end_time = time.time()
        
        conv_out0 = np.reshape(self.output_data[0], self.shapeOut0)
        conv_out1 = np.reshape(self.output_data[1], self.shapeOut1)
        yolo_outputs = [conv_out0, conv_out1]

        boxes, scores, classes = evaluate(yolo_outputs, image_size, self.class_names, self.anchors)

        # 바운딩 박스 시각화 (디버깅용)
        for i, box in enumerate(boxes):
            top_left = (int(box[1]), int(box[0]))
            bottom_right = (int(box[3]), int(box[2]))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        # === 바운딩 박스 기반 차선(직선) 방정식 시각화 추가 ===
        lane_info = self.extract_lane_info_improved(boxes, classes, img)
        h, w = img.shape[:2]
        # 왼쪽 차선 직선 그리기 (slope/intercept 방식)
        if lane_info.left_slope != 0.0:
            try:
                y_bot, y_top = h, 100
                x_bot = lane_info.left_slope * y_bot + lane_info.left_intercept
                x_top = lane_info.left_slope * y_top + lane_info.left_intercept
                cv2.line(img,
                         (int(round(x_bot)), y_bot),
                         (int(round(x_top)), y_top),
                         (255, 0, 0), 2)  # 파란색
            except:
                pass
                
        # 오른쪽 차선 직선 그리기 (slope/intercept 방식)
        if lane_info.right_slope != 0.0:
            try:
                y_bot, y_top = h, 100
                x_bot = lane_info.right_slope * y_bot + lane_info.right_intercept
                x_top = lane_info.right_slope * y_top + lane_info.right_intercept
                cv2.line(img,
                         (int(round(x_bot)), y_bot),
                         (int(round(x_top)), y_top),
                         (0, 0, 255), 2)  # 빨간색
            except:
                pass
        # 기울기 등 텍스트로 표시
        cv2.putText(img, f"Left slope: {lane_info.left_slope:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, f"Right slope: {lane_info.right_slope:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # === 끝 ===

        if use_kanayama:
            # Kanayama 제어기 사용
            lane_info = self.extract_lane_info_improved(boxes, classes, img)
            
            # 기본 조향각 계산
            base_steering_angle, speed = self.kanayama_control(lane_info)
            
            # 강건한 조향각 계산 (히스토리 적용)
            steering_angle = self.get_robust_steering_angle(lane_info, base_steering_angle)
            
            # 차선 정보 디버깅 출력
            print(f"Left: x={lane_info.left_x:.1f}, slope={lane_info.left_slope:.3f}")
            print(f"Right: x={lane_info.right_x:.1f}, slope={lane_info.right_slope:.3f}")
            print(f"Base steering: {base_steering_angle:.2f}°, Final: {steering_angle:.2f}°")
            print(f"History size: {len(self.steering_history)}, No lane count: {self.no_lane_detection_count}")
        else:
            # 기존 방식 사용 (하위 호환성)
            right_lane_center = self.detect_lane_center_x(boxes)
            
            if right_lane_center is None:
                print("차선 중심을 찾을 수 없습니다.")
                # 히스토리에서 평균값 사용
                if len(self.steering_history) > 0:
                    steering_angle = self.get_average_steering()
                    print(f"히스토리 평균 조향각 사용: {steering_angle:.2f}°")
                else:
                    steering_angle = self.default_steering_angle
                speed = self.v_r
            else:
                steering_angle = self.calculate_angle(self.reference_point_x, self.reference_point_y, 
                                                    right_lane_center, self.point_detection_height)
                # 히스토리에 추가
                self.add_steering_to_history(steering_angle)
                speed = self.v_r
        
        # === 최종 주행각도 영상에 표시 ===
        cv2.putText(img, f"Steering Angle: {steering_angle:.2f}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        # === 끝 ===

        return steering_angle, img
