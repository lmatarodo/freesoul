import cv2
import numpy as np
import math
from collections import deque

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
    x1, y1, x2, y2 = box
    roi = binary[y1:y2, x1:x2]          # ① ROI 잘라내기

    # ② ROI 내부 히스토그램으로 초기 x 구함
    hist = np.sum(roi[roi.shape[0]//2:, :], axis=0)
    if np.sum(hist) == 0:  # 히스토그램이 모두 0인 경우
        return None, None
    x_base = np.argmax(hist)            # 왼·오쪽 구분 필요 없으니 하나만
    current_x = x_base

    # ③ 슬라이딩 윈도우
    win_h = roi.shape[0] // n_win
    nz_y, nz_x = roi.nonzero()
    lane_inds = []
    
    for w in range(n_win):
        win_y_low  = roi.shape[0] - (w+1)*win_h
        win_y_high = roi.shape[0] -  w   *win_h
        win_x_low  = current_x - margin
        win_x_high = current_x + margin

        # 윈도우 경계 체크
        win_x_low = max(0, win_x_low)
        win_x_high = min(roi.shape[1], win_x_high)

        good = ((nz_y >= win_y_low) & (nz_y < win_y_high) &
                (nz_x >= win_x_low) & (nz_x < win_x_high)).nonzero()[0]

        if len(good) > minpix:
            lane_inds.append(good)
            current_x = int(np.mean(nz_x[good]))

    lane_inds = np.concatenate(lane_inds) if lane_inds else np.array([])
    if len(lane_inds) == 0:
        return None, None
        
    xs, ys = nz_x[lane_inds], nz_y[lane_inds]

    # ④ 빈 ROI-lane 예외 처리
    if len(xs) < 5:
        return None, None

    # ⑤ ROI 좌표를 전체 이미지 좌표로 변환
    xs_global = xs + x1
    ys_global = ys + y1

    # ⑥ 직선 피팅 (y → x)
    try:
        fit = np.polyfit(ys_global, xs_global, 1)  # slope, intercept
        return fit, (xs_global, ys_global)
    except:
        return None, None

def find_colored_boxes(frame, lower, upper):
    """색상 기반 바운딩 박스 검출"""
    mask = cv2.inRange(frame, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 최소 면적 필터
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))  # (x1, y1, x2, y2) 형식으로 수정
    
    return boxes

class LaneInfo:
    def __init__(self):
        self.left_x = 130
        self.right_x = 130
        self.left_slope = 0.0
        self.right_slope = 0.0
        self.left_intercept = 0.0
        self.right_intercept = 0.0
        self.left_points = None  # 슬라이딩 윈도우 결과 저장용
        self.right_points = None
        
        # ROS 슬라이딩 윈도우 관련 속성들
        self.ros_left_slope = 0.0
        self.ros_right_slope = 0.0
        self.ros_left_fitx = None
        self.ros_right_fitx = None
        self.ros_ploty = None
        self.ros_window_img = None
        self.histogram = None

class LaneVisualizer:
    def __init__(self):
        # 주행 코드와 동일한 파라미터로 수정
        self.v_r = 0.5  # 기준 속도 (m/s)
        self.L = 0.2    # 휠베이스 (m)
        self.K_y = 0.5  # 횡방향 오차 게인
        self.K_phi = 1.0  # 방향각 오차 게인
        self.lane_width = 0.3  # 차로 폭 (m)
        self.default_steering_angle = 0.0

        # 히스토리 관리
        self.steering_history = []
        self.avg_window_size = 5
        
        # 흰색 차선 검출을 위한 HSV 범위
        self.lower_white = np.array([0, 0, 120])      # V 채널을 120으로 완화
        self.upper_white = np.array([180, 80, 255])   # S 채널을 80으로 완화
        
        # 하늘색(왼쪽) 박스 검출을 위한 HSV 범위
        self.cyan_lower = np.array([80, 150, 150])    # H≈80~100, S,V 높음
        self.cyan_upper = np.array([100, 255, 255])
        
        # 빨간색(오른쪽) 박스 검출을 위한 HSV 범위 (빨간색은 HSV에서 0도와 180도 근처에 있음)
        self.red_lower1 = np.array([0, 150, 150])     # 0도 근처
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 150, 150])   # 180도 근처
        self.red_upper2 = np.array([180, 255, 255])

        # 노란색(왼쪽) 박스 검출을 위한 HSV 범위
        self.yellow_lower = np.array([20, 100, 100])   # H≈20~30, S,V 높음
        self.yellow_upper = np.array([30, 255, 255])
        
        # 파란색(오른쪽) 박스 검출을 위한 HSV 범위
        self.blue_lower = np.array([100, 150, 150])    # H≈100~130, S,V 높음
        self.blue_upper = np.array([130, 255, 255])

    def create_binary_image(self, frame):
        """ROS 차선 검출 기법을 적용한 이진화 이미지 생성"""
        # Step 1: Blurring을 통해 노이즈를 제거 (ROS 코드 기법)
        blurred_img = cv2.GaussianBlur(frame, (0, 0), 1)
        
        # Step 2: 색상 필터링 (ROS 코드 기법)
        # HSV 변환 후 흰색 필터링
        hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])      # V 채널을 120으로 완화
        upper_white = np.array([180, 80, 255])   # S 채널을 80으로 완화
        mask_hsv = cv2.inRange(hsv, lower_white, upper_white)
        
        # ROS 코드의 color_filter 방식 적용
        # 흰색 마스크 생성 (더 관대한 범위)
        lower_white_ros = np.array([0, 255, 255])  # ROS 코드 방식
        upper_white_ros = np.array([255, 255, 255])
        white_mask_ros = cv2.inRange(blurred_img, lower_white_ros, upper_white_ros)
        
        # Step 3: 그레이스케일 변환 및 이진화 (ROS 코드 기법)
        gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
        
        # ROS 코드의 고정 임계값 방식 (170)
        _, mask_gray_ros = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        
        # 적응적 임계값 (기존 방식)
        mask_adaptive = cv2.adaptiveThreshold(gray, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        
        # 추가 고정 임계값 (더 낮은 값)
        _, mask_gray_low = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Step 4: 모든 마스크 결합
        mask = cv2.bitwise_or(mask_hsv, white_mask_ros)
        mask = cv2.bitwise_or(mask, mask_gray_ros)
        mask = cv2.bitwise_or(mask, mask_adaptive)
        mask = cv2.bitwise_or(mask, mask_gray_low)

        # Step 5: 모폴로지로 노이즈 정리 (더 부드러운 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # 작은 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # 작은 노이즈 제거
        
        return mask

    def extract_lane_pixels_from_box(self, frame, box):
        """ROS 차선 검출 기법을 적용한 차선 픽셀 추출"""
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]

        # Step 1: Blurring을 통해 노이즈를 제거 (ROS 코드 기법)
        blurred_roi = cv2.GaussianBlur(roi, (0, 0), 1)
        
        # Step 2: 색상 필터링 (ROS 코드 기법)
        # HSV 변환 후 흰색 필터링
        hsv = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 120])      # V 채널을 120으로 완화
        upper_white = np.array([180, 80, 255])   # S 채널을 80으로 완화
        mask_hsv = cv2.inRange(hsv, lower_white, upper_white)
        
        # ROS 코드의 color_filter 방식 적용
        lower_white_ros = np.array([0, 255, 255])  # ROS 코드 방식
        upper_white_ros = np.array([255, 255, 255])
        white_mask_ros = cv2.inRange(blurred_roi, lower_white_ros, upper_white_ros)
        
        # Step 3: 그레이스케일 변환 및 이진화 (ROS 코드 기법)
        gray_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2GRAY)
        
        # ROS 코드의 고정 임계값 방식 (170)
        _, mask_gray_ros = cv2.threshold(gray_roi, 170, 255, cv2.THRESH_BINARY)
        
        # 적응적 임계값 (기존 방식)
        mask_adaptive = cv2.adaptiveThreshold(gray_roi, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        
        # 추가 고정 임계값 (더 낮은 값)
        _, mask_gray_low = cv2.threshold(gray_roi, 120, 255, cv2.THRESH_BINARY)
        
        # Step 4: 모든 마스크 결합
        mask = cv2.bitwise_or(mask_hsv, white_mask_ros)
        mask = cv2.bitwise_or(mask, mask_gray_ros)
        mask = cv2.bitwise_or(mask, mask_adaptive)
        mask = cv2.bitwise_or(mask, mask_gray_low)

        # Step 5: 모폴로지로 노이즈 정리 (더 부드러운 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # 작은 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # 작은 노이즈 제거

        # Step 6: 0이 아닌 픽셀 좌표 모으기 (ROS 코드와 동일한 최소 픽셀 수)
        pts = cv2.findNonZero(mask)  # Nx1x2
        if pts is None or len(pts) < 10:  # ROS 코드와 동일한 minpix=10
            return None
        pts = pts.reshape(-1, 2)      # Nx2, (x, y) 순서
        # ROI 좌표계를 전체 이미지 좌표계로 변환
        pts[:, 0] += x1  # x
        pts[:, 1] += y1  # y
        return pts

    def plot_histogram(self, binary_img):
        """ROS 코드의 plothistogram 함수와 동일한 히스토그램 분석"""
        histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
        midpoint = np.int_(histogram.shape[0]/2)
        leftbase = np.argmax(histogram[:midpoint])
        rightbase = np.argmax(histogram[midpoint:]) + midpoint
        return leftbase, rightbase, histogram

    def slide_window_search_ros_style(self, binary_img, left_current, right_current, nwindows=15, margin=30, minpix=10):
        """ROS 코드의 slide_window_search 함수와 동일한 슬라이딩 윈도우"""
        window_height = np.int_(binary_img.shape[0] / nwindows) 
        nonzero = binary_img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        left_lane = []
        right_lane = []

        out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

        for w in range(nwindows):
            win_y_low = binary_img.shape[0] - (w + 1) * window_height
            win_y_high = binary_img.shape[0] - w * window_height
            win_xleft_low = left_current - margin
            win_xleft_high = left_current + margin
            win_xright_low = right_current - margin
            win_xright_high = right_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                        (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                        (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

            if len(good_left) > minpix:
                left_lane.append(good_left)
                left_current = np.int_(np.mean(nonzero_x[good_left]))  

            if len(good_right) > minpix:
                right_lane.append(good_right)
                right_current = np.int_(np.mean(nonzero_x[good_right]))

        left_lane = np.concatenate(left_lane) if len(left_lane) > 0 else np.array([])
        right_lane = np.concatenate(right_lane) if len(right_lane) > 0 else np.array([])
        leftx = nonzero_x[left_lane] if len(left_lane) > 0 else np.array([])
        lefty = nonzero_y[left_lane] if len(left_lane) > 0 else np.array([])
        rightx = nonzero_x[right_lane] if len(right_lane) > 0 else np.array([])
        righty = nonzero_y[right_lane] if len(right_lane) > 0 else np.array([])

        if len(leftx) > 0 and len(lefty) > 0:
            left_fit = np.polyfit(lefty, leftx, 1)
        else:
            left_fit = [0, 0]

        if len(rightx) > 0 and len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 1)
        else:
            right_fit = [0, 0]

        ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
        left_fitx = left_fit[0] * ploty + left_fit[1]
        right_fitx = right_fit[0] * ploty + right_fit[1]

        for i in range(len(ploty)):
            cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 1, (255, 255, 0), -1)
            cv2.circle(out_img, (int(right_fitx[i]), int(ploty[i])), 1, (255, 255, 0), -1)

        return {'left_fitx': left_fitx, 'left_slope': left_fit[0], 'right_fitx': right_fitx, 'right_slope': right_fit[0], 'ploty': ploty}, out_img

    def draw_sliding_windows(self, frame, binary_img, box, color, n_win=6, margin=20, minpix=20):
        """슬라이딩 윈도우를 시각화하는 함수"""
        x1, y1, x2, y2 = box  # (x1, y1, x2, y2) 형식
        roi = binary_img[y1:y2, x1:x2]
        
        # 히스토그램으로 초기 x 구함
        hist = np.sum(roi[roi.shape[0]//2:, :], axis=0)
        if np.sum(hist) == 0:
            return frame
        
        x_base = np.argmax(hist)
        current_x = x_base
        
        # 슬라이딩 윈도우 그리기
        win_h = roi.shape[0] // n_win
        nz_y, nz_x = roi.nonzero()
        
        for w in range(n_win):
            win_y_low = roi.shape[0] - (w+1)*win_h
            win_y_high = roi.shape[0] - w*win_h
            win_x_low = current_x - margin
            win_x_high = current_x + margin
            
            # 윈도우 경계 체크
            win_x_low = max(0, win_x_low)
            win_x_high = min(roi.shape[1], win_x_high)
            
            # 윈도우 내 픽셀 검출
            good = ((nz_y >= win_y_low) & (nz_y < win_y_high) &
                   (nz_x >= win_x_low) & (nz_x < win_x_high)).nonzero()[0]
            
            # 윈도우 박스 그리기 (전체 이미지 좌표계로 변환)
            win_x1 = x1 + win_x_low
            win_y1 = y1 + win_y_low
            win_x2 = x1 + win_x_high
            win_y2 = y1 + win_y_high
            
            # 윈도우 색상 결정 (픽셀이 있으면 밝게, 없으면 어둡게)
            if len(good) > minpix:
                window_color = color  # 원래 색상
                current_x = int(np.mean(nz_x[good]))  # 중심점 업데이트
            else:
                # 픽셀이 부족한 윈도우는 어둡게 표시
                window_color = tuple(c // 3 for c in color)
            
            # 윈도우 박스 그리기
            cv2.rectangle(frame, (win_x1, win_y1), (win_x2, win_y2), window_color, 1)
            
            # 윈도우 내 픽셀들 표시
            if len(good) > minpix:
                for idx in good:
                    px = int(nz_x[idx] + x1)
                    py = int(nz_y[idx] + y1)
                    cv2.circle(frame, (px, py), 1, color, -1)
        
        return frame

    def extract_lane_info_from_colored_boxes(self, frame):
        """ROS 차선 검출 기법을 적용한 차선 정보 추출"""
        # 1) 이진화 이미지 생성 (ROS 기법 적용)
        binary_img = self.create_binary_image(frame)
        
        # 2) 히스토그램 분석 (ROS 코드 기법)
        left_base, right_base, histogram = self.plot_histogram(binary_img)
        
        # 3) ROS 스타일 슬라이딩 윈도우 검출
        draw_info, ros_window_img = self.slide_window_search_ros_style(binary_img, left_base, right_base)
        
        # 4) 기존 YOLO 박스 검출도 병행
        # 노란색(왼쪽) 박스 검출
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_boxes = find_colored_boxes(hsv, self.yellow_lower, self.yellow_upper)
        
        # 파란색(오른쪽) 박스 검출
        blue_boxes = find_colored_boxes(hsv, self.blue_lower, self.blue_upper)
        
        # 흰색 박스도 검출 (기존)
        white_boxes = find_colored_boxes(frame, self.lower_white, self.upper_white)
        
        # 모든 박스 합치기
        all_boxes = []
        box_types = []  # 박스 타입 추적
        
        for box in yellow_boxes:
            all_boxes.append(box)
            box_types.append("left")
        
        for box in blue_boxes:
            all_boxes.append(box)
            box_types.append("right")
        
        for box in white_boxes:
            all_boxes.append(box)
            box_types.append("white")
        
        lane_info = LaneInfo()
        detected_side = None  # 검출된 박스 방향 플래그
        
        if len(all_boxes) > 0:
            # 가장 큰 박스 선택
            largest_idx = max(range(len(all_boxes)), key=lambda i: (all_boxes[i][2] - all_boxes[i][0]) * (all_boxes[i][3] - all_boxes[i][1]))
            largest_box = all_boxes[largest_idx]
            box_type = box_types[largest_idx]
            x1, y1, x2, y2 = largest_box
            
            # 박스 타입에 따라 방향 결정
            if box_type == "left":
                detected_side = "left"
            elif box_type == "right":
                detected_side = "right"
            else:  # white
                # 흰색 박스는 위치에 따라 방향 결정
                image_center_x = frame.shape[1] // 2
                box_center_x = (x1 + x2) // 2
                
                if box_center_x < image_center_x:
                    detected_side = "left"
                else:
                    detected_side = "right"
            
            # 박스 내부에서 차선 픽셀 추출 (ROS 기법 적용)
            pts = self.extract_lane_pixels_from_box(frame, largest_box)
            
            if pts is not None and len(pts) > 10:
                # 개선된 직선 피팅 방법
                if len(pts) > 1:
                    # y 좌표로 정렬
                    pts_sorted = pts[pts[:, 1].argsort()]
                    
                    # 상위 70% 픽셀만 사용 (더 안정적인 피팅)
                    n_pts = len(pts_sorted)
                    top_pts = pts_sorted[int(0.3 * n_pts):]
                    
                    if len(top_pts) > 1:
                        # 좌표 추출
                        y_coords = top_pts[:, 1].astype(np.float32)
                        x_coords = top_pts[:, 0].astype(np.float32)
                        
                        # 1차: RANSAC으로 이상치 제거 및 직선 피팅
                        if len(y_coords) > 2:
                            try:
                                # RANSAC을 위한 데이터 준비
                                data = np.column_stack([y_coords, x_coords])
                                
                                # RANSAC으로 직선 피팅 (더 엄격한 파라미터)
                                vx, vy, x0, y0 = cv2.fitLine(data, cv2.DIST_L2, 0, 0.01, 0.01)
                                
                                # 직선 방정식 계산: x = slope * y + intercept
                                if abs(vy) > 1e-6:
                                    slope = vx / vy
                                    intercept = x0 - slope * y0
                                else:
                                    slope = 0.0
                                    intercept = x0
                                
                                # 피팅 품질 검증
                                predicted_x = slope * y_coords + intercept
                                residuals = np.abs(x_coords - predicted_x)
                                mean_error = np.mean(residuals)
                                
                                # 오차가 너무 크면 다항식 피팅 시도
                                if mean_error > 5.0:  # 평균 오차가 5픽셀 이상이면
                                    # 2차 다항식 피팅 시도
                                    try:
                                        poly_coeffs = np.polyfit(y_coords, x_coords, 2)
                                        # 2차 다항식의 기울기를 계산 (y의 중간 지점에서)
                                        mid_y = np.mean(y_coords)
                                        slope = 2 * poly_coeffs[0] * mid_y + poly_coeffs[1]
                                        intercept = poly_coeffs[0] * mid_y**2 + poly_coeffs[1] * mid_y + poly_coeffs[2]
                                    except:
                                        # 2차 피팅 실패 시 1차 피팅 사용
                                        coeffs = np.polyfit(y_coords, x_coords, 1)
                                        slope = coeffs[0]
                                        intercept = coeffs[1]
                                else:
                                    # RANSAC 결과가 좋으면 사용
                                    pass
                                
                                # 검출된 방향에 따라 해당하는 차선 정보 설정
                                if detected_side == "left":
                                    lane_info.left_slope = slope
                                    lane_info.left_intercept = intercept
                                    lane_info.left_points = top_pts
                                    lane_info.left_x = x_coords.mean()
                                    lane_info.left_y = y_coords.mean()
                                    lane_info.left_fit_quality = 1.0 / (1.0 + mean_error)  # 품질 점수
                                else:  # right
                                    lane_info.right_slope = slope
                                    lane_info.right_intercept = intercept
                                    lane_info.right_points = top_pts
                                    lane_info.right_x = x_coords.mean()
                                    lane_info.right_y = y_coords.mean()
                                    lane_info.right_fit_quality = 1.0 / (1.0 + mean_error)  # 품질 점수
                                
                                lane_info.confidence = 1.0 / (1.0 + mean_error)
                                
                            except Exception as e:
                                print(f"RANSAC 피팅 실패: {e}")
                                # RANSAC 실패 시 일반 1차 피팅
                                coeffs = np.polyfit(y_coords, x_coords, 1)
                                slope = coeffs[0]
                                intercept = coeffs[1]
                                
                                # 검출된 방향에 따라 해당하는 차선 정보 설정
                                if detected_side == "left":
                                    lane_info.left_slope = slope
                                    lane_info.left_intercept = intercept
                                    lane_info.left_points = top_pts
                                    lane_info.left_x = x_coords.mean()
                                    lane_info.left_y = y_coords.mean()
                                else:  # right
                                    lane_info.right_slope = slope
                                    lane_info.right_intercept = intercept
                                    lane_info.right_points = top_pts
                                    lane_info.right_x = x_coords.mean()
                                    lane_info.right_y = y_coords.mean()
                                
                                lane_info.confidence = 0.5
                        else:
                            # 단순 1차 피팅
                            coeffs = np.polyfit(y_coords, x_coords, 1)
                            slope = coeffs[0]
                            intercept = coeffs[1]
                            
                            # 검출된 방향에 따라 해당하는 차선 정보 설정
                            if detected_side == "left":
                                lane_info.left_slope = slope
                                lane_info.left_intercept = intercept
                                lane_info.left_points = top_pts
                                lane_info.left_x = x_coords.mean()
                                lane_info.left_y = y_coords.mean()
                            else:  # right
                                lane_info.right_slope = slope
                                lane_info.right_intercept = intercept
                                lane_info.right_points = top_pts
                                lane_info.right_x = x_coords.mean()
                                lane_info.right_y = y_coords.mean()
                            
                            lane_info.confidence = 0.3
        
        # ROS 슬라이딩 윈도우 결과는 항상 저장 (참고용)
        lane_info.ros_left_slope = draw_info['left_slope']
        lane_info.ros_right_slope = draw_info['right_slope']
        lane_info.ros_left_fitx = draw_info['left_fitx']
        lane_info.ros_right_fitx = draw_info['right_fitx']
        lane_info.ros_ploty = draw_info['ploty']
        lane_info.ros_window_img = ros_window_img
        lane_info.histogram = histogram
        
        # Kanayama 제어기를 위한 기본값 설정
        if detected_side == "left":
            lane_info.right_x = 130  # 오른쪽 차선 없음 표시
        elif detected_side == "right":
            lane_info.left_x = 130   # 왼쪽 차선 없음 표시
        else:
            # 박스가 검출되지 않았을 때
            lane_info.left_x = 130
            lane_info.right_x = 130
        
        return lane_info, detected_side, binary_img

    def kanayama_control(self, lane_info):
        """주행 코드와 동일한 개선된 Kanayama 제어기"""
        if lane_info.left_x == 130 and lane_info.right_x == 130:
            return self.default_steering_angle, self.v_r
        
        # 이미지 크기 (256x256으로 리사이즈됨)
        image_width = 256
        image_center_x = image_width / 2
        
        # 픽셀→실제 거리 정규화 및 횡방향 오차 계산
        if lane_info.left_x == 130:
            # 왼쪽 차선이 없을 때 - 오른쪽 차선을 따라 주행
            # 오른쪽 차선에서 차로 폭의 절반만큼 왼쪽에 위치하도록
            desired_x = lane_info.right_x - (self.lane_width * 10)  # 픽셀 단위 (1m ≈ 10픽셀 가정)
            norm_x = (desired_x - image_center_x) / (image_width / 2)  # -1 ~ +1 정규화
            lateral_err = -norm_x * (self.lane_width / 2)  # m 단위
            heading_err = lane_info.right_slope
            
        elif lane_info.right_x == 130:
            # 오른쪽 차선이 없을 때 - 왼쪽 차선을 따라 주행
            # 왼쪽 차선에서 차로 폭의 절반만큼 오른쪽에 위치하도록
            desired_x = lane_info.left_x + (self.lane_width * 10)  # 픽셀 단위
            norm_x = (desired_x - image_center_x) / (image_width / 2)  # -1 ~ +1 정규화
            lateral_err = -norm_x * (self.lane_width / 2)  # m 단위
            heading_err = lane_info.left_slope
            
        else:
            # 양쪽 차선이 모두 있을 때 - 차로 중앙 주행
            lane_center_x = (lane_info.left_x + lane_info.right_x) / 2
            norm_x = (lane_center_x - image_center_x) / (image_width / 2)  # -1 ~ +1 정규화
            lateral_err = -norm_x * (self.lane_width / 2)  # m 단위
            heading_err = 0.5 * (lane_info.left_slope + lane_info.right_slope)
        
        heading_err *= -1
        
        # 속도 분모 안정화 (heading_err가 너무 클 때 대응)
        if abs(heading_err) > math.radians(60):  # 60도 이상이면 회피 선회 모드
            # 긴급 회피를 위한 큰 조향각
            steering_angle = 30.0 if heading_err > 0 else -30.0
            v = self.v_r * 0.5  # 속도 감소
        else:
            # 정상 제어 모드
            # 속도 분모 안정화 (최소값 보장)
            v = max(self.v_r * (math.cos(heading_err))**2, 0.1)
        w = self.v_r * (self.K_y * lateral_err + self.K_phi * math.sin(heading_err))
            
            # 조향각 계산
        delta = math.atan2(w * self.L, v)
        steering_angle = math.degrees(delta)
        steering_angle = max(min(steering_angle, 50.0), -50.0)
        
        return steering_angle, v

    def add_steering_to_history(self, steering_angle):
        """조향각을 히스토리에 추가"""
        self.steering_history.append(steering_angle)
        
    def get_average_steering(self, num_frames=None):
        """최근 N개 조향각의 평균 계산"""
        if len(self.steering_history) == 0:
            return self.default_steering_angle
        
        if num_frames is None:
            num_frames = self.avg_window_size
        
        recent_values = list(self.steering_history)[-min(num_frames, len(self.steering_history)):]
        average_steering = sum(recent_values) / len(recent_values)
        
        return average_steering
        
    def get_smoothed_steering(self, current_steering_angle):
        """스무딩을 적용한 조향각 계산"""
        if len(self.steering_history) == 0:
            return current_steering_angle
        
        previous_angle = self.steering_history[-1]
        smoothed_angle = (0.8 * previous_angle + 
                         0.2 * current_steering_angle)
        
        return smoothed_angle
        
    def should_use_history(self, lane_info):
        """히스토리 사용 여부 결정"""
        if lane_info.left_x == 130 and lane_info.right_x == 130:
            return True
        else:
            return False
            
    def get_robust_steering_angle(self, lane_info, current_steering_angle):
        """강건한 조향각 계산 (히스토리 + 스무딩 적용)"""
        if self.should_use_history(lane_info):
            robust_angle = self.get_average_steering()
            return robust_angle
        else:
            smoothed_angle = self.get_smoothed_steering(current_steering_angle)
            self.add_steering_to_history(smoothed_angle)
            return smoothed_angle

    def visualize_frame_from_colored_boxes(self, frame):
        """ROS 차선 검출 기법을 적용한 프레임 시각화"""
        # 차선 정보 추출
        lane_info, detected_side, binary_img = self.extract_lane_info_from_colored_boxes(frame)
        
        # 시각화용 이미지 복사
        vis_img = frame.copy()
        h, w = vis_img.shape[:2]
        
        # 1) 이진화 결과 시각화 (항상 표시)
        binary_vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        
        # 2) 히스토그램 시각화 (ROS 코드 기법) - 항상 표시
        if hasattr(lane_info, 'histogram') and lane_info.histogram is not None:
            hist_vis = np.zeros((200, w, 3), dtype=np.uint8)
            hist_normalized = lane_info.histogram * 200 / np.max(lane_info.histogram)
            for i in range(len(hist_normalized)):
                cv2.line(hist_vis, (i, 200), (i, 200 - int(hist_normalized[i])), (0, 255, 0), 1)
            
            # 히스토그램을 메인 이미지 아래에 추가
            combined_vis = np.vstack([vis_img, hist_vis])
        else:
            combined_vis = vis_img
        
        # 3) ROS 슬라이딩 윈도우 결과 시각화 (항상 표시)
        if hasattr(lane_info, 'ros_window_img') and lane_info.ros_window_img is not None:
            ros_window_resized = cv2.resize(lane_info.ros_window_img, (w//2, h//2))
            # ROS 윈도우 이미지를 우상단에 배치
            combined_vis[0:h//2, w//2:w] = ros_window_resized
        
        # 4) 검출된 방향에 따른 차선 정보 시각화
        if detected_side == "left":
            # 왼쪽 차선만 표시 (노란색)
            # 실제 픽셀들 표시
            if hasattr(lane_info, 'left_points') and lane_info.left_points is not None:
                for pt in lane_info.left_points:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(combined_vis, (x, y), 1, (0, 255, 255), -1)  # 노란색 픽셀
            
            # 왼쪽 차선 직선 그리기 (노란색)
            if hasattr(lane_info, 'left_slope') and hasattr(lane_info, 'left_intercept'):
                slope = lane_info.left_slope
                intercept = lane_info.left_intercept
                
                if np.isfinite(slope) and np.isfinite(intercept):
                    # 직선의 시작점과 끝점 계산
                    y1 = 0
                    y2 = h
                    x1 = int(slope * y1 + intercept)
                    x2 = int(slope * y2 + intercept)
                    
                    # 유효한 좌표인지 확인
                    if 0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h:
                        cv2.line(combined_vis, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 노란색 직선
                        
                        # 피팅 품질 정보 표시
                        if hasattr(lane_info, 'left_fit_quality'):
                            quality_text = f"Fit Quality: {lane_info.left_fit_quality:.3f}"
                            cv2.putText(combined_vis, quality_text, (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        elif detected_side == "right":
            # 오른쪽 차선만 표시 (파란색)
            # 실제 픽셀들 표시
            if hasattr(lane_info, 'right_points') and lane_info.right_points is not None:
                for pt in lane_info.right_points:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(combined_vis, (x, y), 1, (255, 0, 0), -1)  # 파란색 픽셀
            
            # 오른쪽 차선 직선 그리기 (파란색)
            if hasattr(lane_info, 'right_slope') and hasattr(lane_info, 'right_intercept'):
                slope = lane_info.right_slope
                intercept = lane_info.right_intercept
                
                if np.isfinite(slope) and np.isfinite(intercept):
                    # 직선의 시작점과 끝점 계산
                    y1 = 0
                    y2 = h
                    x1 = int(slope * y1 + intercept)
                    x2 = int(slope * y2 + intercept)
                    
                    # 유효한 좌표인지 확인
                    if 0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h:
                        cv2.line(combined_vis, (x1, y1), (x2, y2), (255, 0, 0), 3)  # 파란색 직선
                        
                        # 피팅 품질 정보 표시
                        if hasattr(lane_info, 'right_fit_quality'):
                            quality_text = f"Fit Quality: {lane_info.right_fit_quality:.3f}"
                            cv2.putText(combined_vis, quality_text, (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 5) 차선 정보 텍스트 표시
        info_text = []
        if detected_side == "left":
            if hasattr(lane_info, 'left_slope'):
                try:
                    slope_val = float(lane_info.left_slope)
                except Exception:
                    slope_val = lane_info.left_slope if isinstance(lane_info.left_slope, (int, float)) else 0.0
                info_text.append(f"Left Slope: {slope_val:.3f}")
        elif detected_side == "right":
            if hasattr(lane_info, 'right_slope'):
                try:
                    slope_val = float(lane_info.right_slope)
                except Exception:
                    slope_val = lane_info.right_slope if isinstance(lane_info.right_slope, (int, float)) else 0.0
                info_text.append(f"Right Slope: {slope_val:.3f}")
        
        if hasattr(lane_info, 'confidence'):
            try:
                conf_val = float(lane_info.confidence)
            except Exception:
                conf_val = lane_info.confidence if isinstance(lane_info.confidence, (int, float)) else 0.0
            info_text.append(f"Conf: {conf_val:.2f}")
        
        # 검출된 방향 표시
        if detected_side:
            info_text.append(f"Detected: {detected_side.upper()}")
        else:
            info_text.append("No lane detected")
        
        for i, text in enumerate(info_text):
            cv2.putText(combined_vis, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 6) 조향각 계산 및 표시
        if detected_side and hasattr(lane_info, 'left_x') and hasattr(lane_info, 'right_x'):
            steering_angle, speed = self.kanayama_control(lane_info)
            
            # 조향각 정보를 더 자세히 표시
            steering_text = []
            steering_text.append(f"Steering: {steering_angle:.1f}°")
            steering_text.append(f"Speed: {speed:.2f} m/s")
            
            # 조향각 히스토리 정보
            if len(self.steering_history) > 0:
                avg_steering = self.get_average_steering()
                steering_text.append(f"Avg: {avg_steering:.1f}°")
                steering_text.append(f"History: {len(self.steering_history)}")
            
            # 차선 정보 (Kanayama 제어기에서 사용하는 값들)
            if hasattr(lane_info, 'left_x') and hasattr(lane_info, 'right_x'):
                try:
                    left_x_val = float(lane_info.left_x)
                    right_x_val = float(lane_info.right_x)
                except Exception:
                    left_x_val = lane_info.left_x if isinstance(lane_info.left_x, (int, float)) else 0.0
                    right_x_val = lane_info.right_x if isinstance(lane_info.right_x, (int, float)) else 0.0
                steering_text.append(f"Left X: {left_x_val:.1f}")
                steering_text.append(f"Right X: {right_x_val:.1f}")
            
            # 조향각 텍스트를 화면에 표시
            for i, text in enumerate(steering_text):
                cv2.putText(combined_vis, text, (10, h - 100 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            # 차선이 검출되지 않았을 때
            cv2.putText(combined_vis, "No lane detected", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 7) 이진화 결과를 좌하단에 표시
        binary_resized = cv2.resize(binary_vis, (w//3, h//3))
        combined_vis[h-h//3:h, 0:w//3] = binary_resized
        
        return combined_vis, lane_info

def main():
    visualizer = LaneVisualizer()
    cap = cv2.VideoCapture('../result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    slow_fps = original_fps / 18
    print(f"원본 FPS: {original_fps}, 느린 재생 FPS: {slow_fps}")
    
    # 이진화 결과를 별도 창으로 표시할지 여부
    show_binary = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter('lane_prediction_visualized.mp4', fourcc, slow_fps, (int(w*1.5), int(h*1.5)))
        
        vis, lane_info = visualizer.visualize_frame_from_colored_boxes(frame)
        vis_large = cv2.resize(vis, (int(w*1.5), int(h*1.5)))
        cv2.imshow('Lane Visualization', vis_large)
        
        out.write(vis_large)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            show_binary = not show_binary
            if not show_binary:
                cv2.destroyWindow('Binary Image')
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print('완료: lane_prediction_visualized.mp4 (색상 박스 기반)')

if __name__ == '__main__':
    main()