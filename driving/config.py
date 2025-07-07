# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import numpy as np

MOTOR_ADDRESSES = {
    'motor_0': 0x00A0000000,
    'motor_1': 0x00A0010000,
    'motor_2': 0x00A0020000,
    'motor_3': 0x00A0030000,
    'motor_4': 0x00A0040000,
    'motor_5': 0x00A0050000
}

ADDRESS_RANGE = 0x10000

# YOLO configurations
anchor_list = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
anchors = np.array(anchor_list).reshape(-1, 2)
classes_path = "../xmodel/lane_class.txt"

# Kanayama 제어기 파라미터
KANAYAMA_CONFIG = {
    'K_y': 0.3,        # 횡방향 오차 게인
    'K_phi': 0.9,      # 방향각 오차 게인
    'L': 0.5,          # 휠베이스 (m)
    'lane_width': 3.5, # 차선 폭 (m)
    'v_r': 30.0        # 기준 속도
}

# 히스토리 관리 설정
HISTORY_CONFIG = {
    'max_history_size': 10,      # 최대 히스토리 크기
    'avg_window_size': 5,        # 평균 계산에 사용할 프레임 수
    'max_no_lane_frames': 5,     # 최대 차선 미검출 프레임 수
    'default_steering_angle': 0.0,  # 기본 조향각 (직진)
    'smoothing_factor': 0.2      # 스무딩 팩터 (0~1, 높을수록 부드러움)
}

# 기존 제어 파라미터 (하위 호환성)
k_p = 30/25  # 속도 기반 비례 게인
k_para = 20
k_lateral = 5


