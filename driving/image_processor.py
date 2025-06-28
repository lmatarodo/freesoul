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

class LaneInfo:
    """차선 정보를 저장하는 클래스"""
    def __init__(self):
        self.left_x = 130  # 왼쪽 차선 x좌표 (기본값: 차선 없음)
        self.right_x = 130  # 오른쪽 차선 x좌표 (기본값: 차선 없음)
        self.left_slope = 0.0  # 왼쪽 차선 기울기
        self.right_slope = 0.0  # 오른쪽 차선 기울기

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

    def extract_lane_pixels(self, gray_img, box):
        """바운딩 박스 내의 차선 픽셀들을 추출 (개선된 방식)"""
        y1, x1, y2, x2 = box
        
        # 바운딩 박스 영역 추출
        roi = gray_img[int(y1):int(y2), int(x1):int(x2)]
        
        # 참고 코드와 동일한 이진화 임계값 사용
        _, binary = cv2.threshold(roi, 170, 255, cv2.THRESH_BINARY)
        
        # 차선 픽셀들의 좌표 찾기 (참고 코드와 동일한 방식)
        nonzero = binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        # 원본 이미지 좌표로 변환
        lane_pixels = []
        for i in range(len(nonzero_y)):
            orig_y = int(y1) + nonzero_y[i]
            orig_x = int(x1) + nonzero_x[i]
            lane_pixels.append((orig_y, orig_x))
        
        return lane_pixels

    def extract_lane_info(self, xyxy_results, classes_results, processed_img):
        """바운딩 박스에서 좌우 차선 정보를 추출 (픽셀 기반 직선 피팅)"""
        lane_info = LaneInfo()
        
        if len(xyxy_results) == 0:
            return lane_info
        
        left_lanes = []
        right_lanes = []
        
        # 이미지를 그레이스케일로 변환 (이미 처리된 이미지 사용)
        if len(processed_img.shape) == 3:
            gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = processed_img
        
        for i, box in enumerate(xyxy_results):
            y1, x1, y2, x2 = box
            
            # 바운딩 박스 내의 차선 픽셀들을 찾기
            lane_pixels = self.extract_lane_pixels(gray_img, box)
            
            if len(lane_pixels) < 10:  # 최소 픽셀 수 확인
                continue
            
            # 픽셀 좌표를 y, x 형태로 변환 (참고 코드와 동일)
            pixels_y = [p[0] for p in lane_pixels]  # y 좌표
            pixels_x = [p[1] for p in lane_pixels]  # x 좌표
            
            # 직선 피팅 수행 (참고 코드와 동일한 방식)
            try:
                fit = np.polyfit(pixels_y, pixels_x, 1)  # x = slope * y + intercept
                slope = fit[0]  # 기울기
                intercept = fit[1]  # y절편
                
                # 차선의 중심점 계산 (이미지 하단 y=255에서의 x 좌표)
                # 참고 코드와 동일한 방식으로 계산
                center_x = slope * 255 + intercept
                
                # 클래스 정보를 기반으로 좌우 차선 분류
                class_id = classes_results[i] if i < len(classes_results) else 0
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else ""
                
                if "left" in class_name.lower():
                    left_lanes.append((center_x, slope))
                elif "right" in class_name.lower():
                    right_lanes.append((center_x, slope))
                else:
                    # 클래스 이름이 명확하지 않은 경우, 기존 방식으로 분류 (fallback)
                    image_center_x = 128
                    if center_x < image_center_x:
                        left_lanes.append((center_x, slope))
                    else:
                        right_lanes.append((center_x, slope))
                        
            except (np.RankWarning, ValueError):
                # 직선 피팅 실패 시 기존 방식 사용
                center_x = (x1 + x2) / 2
                slope = (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else 0
                
                class_id = classes_results[i] if i < len(classes_results) else 0
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else ""
                
                if "left" in class_name.lower():
                    left_lanes.append((center_x, slope))
                elif "right" in class_name.lower():
                    right_lanes.append((center_x, slope))
        
        # 왼쪽 차선 정보 설정
        if left_lanes:
            # 가장 오른쪽에 있는 왼쪽 차선 선택
            left_lanes.sort(key=lambda x: x[0], reverse=True)
            lane_info.left_x = left_lanes[0][0]
            lane_info.left_slope = left_lanes[0][1]
        
        # 오른쪽 차선 정보 설정
        if right_lanes:
            # 가장 왼쪽에 있는 오른쪽 차선 선택
            right_lanes.sort(key=lambda x: x[0])
            lane_info.right_x = right_lanes[0][0]
            lane_info.right_slope = right_lanes[0][1]
        
        return lane_info
    
    def kanayama_control(self, lane_info):
        """Kanayama 제어기를 사용한 조향각 계산 (히스토리 기능 포함)"""
        if lane_info.left_x == 130 and lane_info.right_x == 130:
            print("차선을 찾을 수 없습니다.")
            # 히스토리에서 평균값 사용
            if len(self.steering_history) > 0:
                avg_steering = self.get_average_steering()
                print(f"히스토리 평균 조향각 사용: {avg_steering:.2f}°")
                return avg_steering, self.v_r
            else:
                return self.default_steering_angle, self.v_r
        
        # 횡방향 오차와 방향각 오차 계산
        if lane_info.left_x == 130:
            # 왼쪽 차선이 없을 때
            lateral_err = -(0.5 - (lane_info.right_x / 150.0)) * self.lane_width
            heading_err = lane_info.right_slope
        elif lane_info.right_x == 130:
            # 오른쪽 차선이 없을 때
            lateral_err = (0.5 - (lane_info.left_x / 150.0)) * self.lane_width
            heading_err = lane_info.left_slope
        else:
            # 양쪽 차선이 모두 있을 때
            lateral_err = -(lane_info.left_x / (lane_info.left_x + lane_info.right_x) - 0.5) * self.lane_width
            heading_err = 0.5 * (lane_info.left_slope + lane_info.right_slope)
        
        heading_err *= -1  # 방향 보정
        
        # 각속도 계산
        v = self.v_r * (math.cos(heading_err))**2
        w = self.v_r * (self.K_y * lateral_err + self.K_phi * math.sin(heading_err))
        
        # 조향각 계산 (라디안 -> 각도)
        delta = math.atan2(w * self.L, v)
        steering_angle = math.degrees(delta)
        
        # 조향각 제한 (-50 ~ 50도)
        steering_angle = max(min(steering_angle, 50.0), -50.0)
        
        return steering_angle, v

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

        if use_kanayama:
            # Kanayama 제어기 사용
            lane_info = self.extract_lane_info(boxes, classes, img)
            
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
        
        return steering_angle, img
