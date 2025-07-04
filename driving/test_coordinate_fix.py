#!/usr/bin/env python3
"""
좌표계 수정 테스트 스크립트
"""

import cv2
import numpy as np
import sys
import os

# 상위 디렉토리의 모듈을 import하기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driving.visualize_lane_prediction import find_colored_boxes, LaneVisualizer

def test_coordinate_consistency():
    """좌표계 일관성 테스트"""
    print("좌표계 일관성 테스트")
    
    # 테스트 이미지 생성 (256x256)
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 하늘색 박스 그리기 (테스트용)
    cv2.rectangle(test_img, (50, 100), (150, 200), (255, 255, 0), -1)
    # 빨간색 박스 그리기 (테스트용)
    cv2.rectangle(test_img, (180, 100), (220, 200), (0, 0, 255), -1)
    
    # find_colored_boxes 함수 테스트
    lower_cyan = np.array([200, 200, 0])
    upper_cyan = np.array([255, 255, 80])
    cyan_boxes = find_colored_boxes(test_img, lower_cyan, upper_cyan)
    
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([80, 80, 255])
    red_boxes = find_colored_boxes(test_img, lower_red, upper_red)
    
    print(f"하늘색 박스: {cyan_boxes}")
    print(f"빨간색 박스: {red_boxes}")
    
    # 좌표 형식 확인
    for i, box in enumerate(cyan_boxes):
        x1, y1, x2, y2 = box
        print(f"하늘색 박스 {i}: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
        print(f"  - 너비: {x2-x1}, 높이: {y2-y1}")
    
    for i, box in enumerate(red_boxes):
        x1, y1, x2, y2 = box
        print(f"빨간색 박스 {i}: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
        print(f"  - 너비: {x2-x1}, 높이: {y2-y1}")
    
    # 시각화 테스트
    visualizer = LaneVisualizer()
    vis = visualizer.visualize_frame_from_colored_boxes(test_img)
    
    # 결과 저장
    cv2.imwrite('test_coordinate_consistency.jpg', vis)
    print("테스트 결과 저장: test_coordinate_consistency.jpg")
    
    # 결과 표시
    cv2.imshow('Coordinate Consistency Test', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_sliding_window_visualization():
    """슬라이딩 윈도우 시각화 테스트"""
    print("슬라이딩 윈도우 시각화 테스트")
    
    # 테스트 이미지 생성 (차선이 있는 이미지 시뮬레이션)
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 차선 그리기 (흰색)
    cv2.line(test_img, (100, 200), (120, 100), (255, 255, 255), 5)  # 왼쪽 차선
    cv2.line(test_img, (180, 200), (200, 100), (255, 255, 255), 5)  # 오른쪽 차선
    
    # 하늘색 박스 그리기 (왼쪽 차선 영역)
    cv2.rectangle(test_img, (80, 80), (140, 220), (255, 255, 0), 2)
    # 빨간색 박스 그리기 (오른쪽 차선 영역)
    cv2.rectangle(test_img, (160, 80), (220, 220), (0, 0, 255), 2)
    
    # 시각화 테스트
    visualizer = LaneVisualizer()
    vis = visualizer.visualize_frame_from_colored_boxes(test_img)
    
    # 결과 저장
    cv2.imwrite('test_sliding_window_visualization.jpg', vis)
    print("테스트 결과 저장: test_sliding_window_visualization.jpg")
    
    # 결과 표시
    cv2.imshow('Sliding Window Visualization Test', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_line_clipping():
    """직선 클리핑 테스트"""
    print("직선 클리핑 테스트")
    
    # 테스트 이미지 생성
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 박스 그리기
    box = (50, 100, 150, 200)  # (x1, y1, x2, y2)
    x1, y1, x2, y2 = box
    cv2.rectangle(test_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    # 박스 밖으로 나가는 직선 그리기 (클리핑 전)
    slope = 0.5
    intercept = 50
    
    # 박스 하단과 상단에서 x 좌표 계산
    x_bot = slope * y2 + intercept
    x_top = slope * y1 + intercept
    
    print(f"클리핑 전: x_bot={x_bot:.1f}, x_top={x_top:.1f}")
    print(f"박스 범위: x1={x1}, x2={x2}")
    
    # 박스 범위 밖으로 벗어난 x 값은 잘라내기
    x_bot_clipped = np.clip(x_bot, x1, x2)
    x_top_clipped = np.clip(x_top, x1, x2)
    
    print(f"클리핑 후: x_bot={x_bot_clipped:.1f}, x_top={x_top_clipped:.1f}")
    
    # 클리핑 전 직선 (빨간색)
    cv2.line(test_img, (int(x_bot), y2), (int(x_top), y1), (0, 0, 255), 2)
    
    # 클리핑 후 직선 (파란색)
    cv2.line(test_img, (int(x_bot_clipped), y2), (int(x_top_clipped), y1), (255, 0, 0), 3)
    
    # 결과 저장
    cv2.imwrite('test_line_clipping.jpg', test_img)
    print("테스트 결과 저장: test_line_clipping.jpg")
    
    # 결과 표시
    cv2.imshow('Line Clipping Test', test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """메인 함수"""
    print("좌표계 수정 테스트 시작")
    
    # 1. 좌표계 일관성 테스트
    test_coordinate_consistency()
    
    # 2. 슬라이딩 윈도우 시각화 테스트
    test_sliding_window_visualization()
    
    # 3. 직선 클리핑 테스트
    test_line_clipping()
    
    print("모든 테스트 완료!")

if __name__ == '__main__':
    main() 