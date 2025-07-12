# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import math
import time
import numpy as np
from threading import Lock
from enum import Enum
from config import ULTRASONIC_ADDRESSES, ADDRESS_RANGE

class ParkingPhase(Enum):
    """주차 단계 열거형"""
    WAITING = 0
    INITIAL_FORWARD = 1
    FIRST_STOP = 2
    LEFT_TURN_FORWARD = 3
    SECOND_STOP = 4
    RIGHT_TURN_BACKWARD = 5
    STRAIGHT_BACKWARD = 6
    ALIGNMENT = 7
    POSITION_CHECK = 8
    CORRECTION = 9
    POST_CORRECTION_BACKWARD = 10
    PARKING_COMPLETE_STOP = 11
    FINAL_FORWARD = 12
    COMPLETED = 13

class ParkingSystemController:
    """자율주차 시스템 컨트롤러"""
    
    def __init__(self, motor_controller, ultrasonic_sensors=None):
        """
        주차 시스템 컨트롤러 초기화
        
        Args:
            motor_controller: 모터 제어기 인스턴스
            ultrasonic_sensors: 초음파 센서 딕셔너리 (선택사항)
        """
        self.motor_controller = motor_controller
        self.ultrasonic_sensors = ultrasonic_sensors or {}
        
        # 초음파 센서 매핑 (센서 위치별)
        self.sensor_mapping = {
            "front_right": "ultrasonic_0",    # 전방 우측
            "middle_left": "ultrasonic_1",    # 중간 좌측
            "middle_right": "ultrasonic_2",   # 중간 우측
            "rear_left": "ultrasonic_3",      # 후방 좌측
            "rear_right": "ultrasonic_4"      # 후방 우측
        }
        
        # 주차 상태 변수
        self.current_phase = ParkingPhase.WAITING
        self.status_message = "대기 중..."
        self.is_parking_active = False
        self.parking_completed = False
        
        # 센서 데이터
        self.sensor_distances = {
            "front_right": 100,
            "middle_left": 100,
            "middle_right": 100,
            "rear_left": 100,
            "rear_right": 100
        }
        
        # 이전 센서 값 (변화 감지용)
        self.previous_distances = {
            "front_right": -1,
            "middle_right": -1,
            "rear_right": -1
        }
        
        # 센서 감지 상태 플래그
        self.sensor_flags = {
            "front_right": False,
            "middle_right": False,
            "rear_right": False
        }
        
        # 주차 단계별 상태 변수
        self.phase_states = {
            'initial_forward_started': False,
            'first_stop_completed': False,
            'left_turn_started': False,
            'second_stop_completed': False,
            'right_turn_started': False,
            'backward_completed': False,
            'alignment_completed': False,
            'straight_backward_started': False,
            'correction_started': False,
            'correction_completed': False,
            'post_correction_backward_started': False,
            'parking_completion_stop_started': False,
            'parking_completion_forward_started': False,
            'right_turn_after_increase_started': False
        }
        
        # 시간 관련 변수
        self.phase_start_time = None
        self.correction_start_time = None
        self.backward_start_time = None
        self.straight_backward_start_time = None
        self.post_correction_backward_start_time = None
        self.parking_completion_stop_start_time = None
        self.right_turn_after_increase_start_time = None
        self.additional_backward_start_time = None  # 추가 후진 시작 시간
        
        # 수정 관련 변수
        self.correction_direction = 0  # 수정 방향 (1: 우회전, -1: 좌회전, 0: 미정)
        
        # 주차 설정
        self.parking_config = {
            'forward_speed': 30,      # 전진 속도 (0-100)
            'backward_speed': 25,     # 후진 속도 (0-100)
            'steering_speed': 50,     # 조향 속도 (0-100)
            'left_turn_angle': -20,   # 좌회전 각도
            'right_turn_angle': 13,   # 우회전 각도
            'correction_angle': 15,   # 수정 조향 각도
            'stop_distance': 40,      # 정지 거리 (cm)
            'alignment_tolerance': 3, # 정렬 허용 오차 (cm)
            'correction_threshold': 10, # 수정 임계값 (cm)
            'straight_backward_duration': 0.3, # 정방향 후진 시간 (초)
            'correction_duration': 2.0, # 수정 시간 (초)
            'parking_stop_duration': 2.0, # 주차 완료 정지 시간 (초)
            'right_turn_duration': 1.5,  # 우회전 시간 (초)
            'additional_backward_duration': 0.5,  # 추가 후진 시간 (초)
            'final_right_turn_angle': 20  # 최종 우회전 각도
        }
        
        # 스레드 안전을 위한 락
        self._lock = Lock()
        
    def start_parking(self):
        """주차 시작"""
        with self._lock:
            self.is_parking_active = True
            self.parking_completed = False
            self.current_phase = ParkingPhase.WAITING
            self.status_message = "주차 시작..."
            self._reset_phase_states() 
            print("🚗 주차 시스템 시작")
    
    def stop_parking(self):
        """주차 중지"""
        with self._lock:
            self.is_parking_active = False
            self.motor_controller.reset_motor_values()
            self.status_message = "주차 중지됨"
            print("🛑 주차 시스템 중지")
    
    def _reset_phase_states(self):
        """단계별 상태 초기화"""
        for key in self.phase_states:
            self.phase_states[key] = False
        self.phase_start_time = None
        self.additional_backward_start_time = None
        self.correction_direction = 0  # 수정 방향 초기화
    
    def update_sensor_data(self, sensor_data):
        """
        센서 데이터 업데이트
        
        Args:
            sensor_data: 센서 거리 데이터 딕셔너리
        """
        with self._lock:
            self.sensor_distances.update(sensor_data)
    
    def read_ultrasonic_sensors(self):
        """
        초음파 센서에서 실제 데이터 읽기
        
        Returns:
            dict: 센서 위치별 거리 데이터
        """
        sensor_data = {}
        
        try:
            for sensor_name, ultrasonic_id in self.sensor_mapping.items():
                if ultrasonic_id in self.ultrasonic_sensors:
                    # 실제 센서에서 데이터 읽기
                    distance = self._read_single_sensor(ultrasonic_id)
                    sensor_data[sensor_name] = distance
                else:
                    # 센서가 없으면 기본값 사용
                    sensor_data[sensor_name] = 100
            
            return sensor_data
            
        except Exception as e:
            print(f"센서 읽기 오류: {e}")
            # 오류 시 기본값 반환
            return {
                "front_right": 100,
                "middle_left": 100,
                "middle_right": 100,
                "rear_left": 100,
                "rear_right": 100
            }
    
    def _read_single_sensor(self, sensor_id):
        """
        단일 센서에서 데이터 읽기
        
        Args:
            sensor_id: 센서 ID
            
        Returns:
            float: 센서 거리 (cm)
        """
        try:
            if sensor_id in self.ultrasonic_sensors:
                sensor = self.ultrasonic_sensors[sensor_id]
                # 실제 센서 읽기 구현 (MMIO를 통한 하드웨어 접근)
                # 센서에서 거리 데이터 읽기 (예: 0x00 주소에서 4바이트 읽기)
                try:
                    distance_raw = sensor.read(0x00, 4)  # 4바이트 읽기
                    distance = int.from_bytes(distance_raw, byteorder='little') / 100.0  # cm 단위로 변환
                    return max(0, min(500, distance))  # 0~500cm 범위로 제한
                except Exception as read_error:
                    print(f"센서 {sensor_id} 하드웨어 읽기 오류: {read_error}")
                    return 100  # 기본값 반환
            else:
                return 100
        except Exception as e:
            print(f"센서 {sensor_id} 읽기 오류: {e}")
            return 100
    
    def _get_sensor_distance(self, sensor_name):
        """센서 거리 가져오기"""
        return self.sensor_distances.get(sensor_name, 100)
    
    def _check_sensor_detection(self):
        """센서 감지 상태 확인 (첫 번째 정지 조건)"""
        current_distances = {
            "front_right": self._get_sensor_distance("front_right"),
            "middle_right": self._get_sensor_distance("middle_right"),
            "rear_right": self._get_sensor_distance("rear_right")
        }
        
        # 각 센서별로 개별적으로 작아졌다가 커지는지 확인
        for sensor_name in ["front_right", "middle_right", "rear_right"]:
            current = current_distances[sensor_name]
            previous = self.previous_distances[sensor_name]
            
            # 아직 감지되지 않은 센서만 확인
            if not self.sensor_flags[sensor_name] and previous > 0:
                if current > previous + 5:  # 5cm 이상 증가
                    self.sensor_flags[sensor_name] = True
                    print(f"✅ {sensor_name} 센서 감지 완료!")
        
        # 모든 우측 센서가 한 번씩 작아졌다가 커졌는지 확인
        if all(self.sensor_flags.values()) and not self.phase_states['first_stop_completed']:
            self.status_message = "모든 우측 센서 감지 완료! 정지 신호!"
            return True
        
        self.previous_distances = current_distances.copy()
        return False
    
    def _check_second_stop_condition(self):
        """두 번째 정지 조건 확인"""
        rear_right_current = self._get_sensor_distance("rear_right")
        
        if rear_right_current > 0 and self.previous_distances["rear_right"] > 0:
            if rear_right_current > self.previous_distances["rear_right"] + 10:
                self.status_message = "두 번째 정지 신호 감지!"
                return True
        
        return False
    
    def _check_backward_completion(self):
        """후진 완료 조건 확인"""
        front_right_distance = self._get_sensor_distance("front_right")
        
        if front_right_distance <= self.parking_config['stop_distance']:
            self.status_message = "후진 완료!"
            return True
        
        return False
    
    def _check_alignment_completion(self):
        """차량 정렬 완료 조건 확인"""
        front_right_distance = self._get_sensor_distance("front_right")
        rear_right_distance = self._get_sensor_distance("rear_right")
        
        # 센서 값이 유효한지 확인
        if front_right_distance <= 0 or rear_right_distance <= 0:
            return False
        
        # front_right와 rear_right 값의 차이 계산
        distance_diff = front_right_distance - rear_right_distance
        tolerance = self.parking_config['alignment_tolerance']
        
        # 차량 정렬 상태 확인
        if abs(distance_diff) <= tolerance:
            self.status_message = "차량 정렬 완료! 주차 완료!"
            return True
        else:
            # 차량 정렬을 위한 조향 조정 - 시뮬레이션과 일치하도록 5도 사용
            if distance_diff > 0:
                # front_right가 더 크면 왼쪽으로 조향
                self._set_steering_angle(-5)  # 시뮬레이션과 일치
                self.status_message = "왼쪽 조향으로 정렬 중..."
            else:
                # rear_right가 더 크면 오른쪽으로 조향
                self._set_steering_angle(5)  # 시뮬레이션과 일치
                self.status_message = "오른쪽 조향으로 정렬 중..."
            
            return False
    
    def _check_position_correction_needed(self):
        """위치 수정 필요 여부 확인"""
        middle_right_distance = self._get_sensor_distance("middle_right")
        middle_left_distance = self._get_sensor_distance("middle_left")
        
        # 센서 값이 유효한지 확인
        if middle_right_distance <= 0 and middle_left_distance <= 0:
            self.status_message = "주차 완료!"
            return False
        
        # middle_right와 middle_left 값의 차이 계산
        distance_diff = abs(middle_right_distance - middle_left_distance)
        correction_threshold = self.parking_config['correction_threshold']
        
        if distance_diff >= correction_threshold:
            if middle_right_distance > middle_left_distance:
                self.status_message = "좌측으로 치우침! 수정 필요!"
            else:
                self.status_message = "우측으로 치우침! 수정 필요!"
            return True
        else:
            self.status_message = "주차 완료!"
            return False
    
    def _check_time_elapsed(self, start_time, duration):
        """시간 경과 확인"""
        if start_time is None:
            return False
        return (time.time() - start_time) >= duration
    
    def _set_phase(self, phase):
        """단계 설정"""
        self.current_phase = phase
        self.phase_start_time = time.time()
        print(f"🔄 단계 변경: {phase.name}")
    
    def _stop_vehicle(self):
        """차량 정지"""
        self.motor_controller.reset_motor_values()
    
    def _move_forward(self, speed=None):
        """전진"""
        speed = speed or self.parking_config['forward_speed']
        self.motor_controller.left_speed = speed
        self.motor_controller.right_speed = speed
    
    def _move_backward(self, speed=None):
        """후진"""
        speed = speed or self.parking_config['backward_speed']
        self.motor_controller.left_speed = -speed
        self.motor_controller.right_speed = -speed
    
    def _turn_left(self):
        """좌회전 - 설정된 각도로 조향"""
        angle = self.parking_config['left_turn_angle']
        print(f"[PARKING_DEBUG] 좌회전: {angle}도")
        self.motor_controller.control_motors(angle, control_mode=1)
    
    def _turn_right(self):
        """우회전 - 설정된 각도로 조향"""
        angle = self.parking_config['right_turn_angle']
        print(f"[PARKING_DEBUG] 우회전: {angle}도")
        self.motor_controller.control_motors(angle, control_mode=1)
    
    def _straight_steering(self):
        """직진 조향 - 0도로 조향"""
        print(f"[PARKING_DEBUG] 직진: 0도")
        self.motor_controller.control_motors(0.0, control_mode=1)
    
    def _set_steering_angle(self, angle):
        """특정 각도로 조향 설정"""
        print(f"[PARKING_DEBUG] 조향각 설정: {angle}도")
        self.motor_controller.control_motors(angle, control_mode=1)
    
    def execute_parking_cycle(self):
        """주차 사이클 실행"""
        if not self.is_parking_active:
            return
        
        with self._lock:
            try:
                # 실제 센서 데이터 읽기
                sensor_data = self.read_ultrasonic_sensors()
                self.update_sensor_data(sensor_data)
                
                if self.current_phase == ParkingPhase.WAITING:
                    self._execute_waiting_phase()
                elif self.current_phase == ParkingPhase.INITIAL_FORWARD:
                    self._execute_initial_forward_phase()
                elif self.current_phase == ParkingPhase.FIRST_STOP:
                    self._execute_first_stop_phase()
                elif self.current_phase == ParkingPhase.LEFT_TURN_FORWARD:
                    self._execute_left_turn_forward_phase()
                elif self.current_phase == ParkingPhase.SECOND_STOP:
                    self._execute_second_stop_phase()
                elif self.current_phase == ParkingPhase.RIGHT_TURN_BACKWARD:
                    self._execute_right_turn_backward_phase()
                elif self.current_phase == ParkingPhase.STRAIGHT_BACKWARD:
                    self._execute_straight_backward_phase()
                elif self.current_phase == ParkingPhase.ALIGNMENT:
                    self._execute_alignment_phase()
                elif self.current_phase == ParkingPhase.POSITION_CHECK:
                    self._execute_position_check_phase()
                elif self.current_phase == ParkingPhase.CORRECTION:
                    self._execute_correction_phase()
                elif self.current_phase == ParkingPhase.POST_CORRECTION_BACKWARD:
                    self._execute_post_correction_backward_phase()
                elif self.current_phase == ParkingPhase.PARKING_COMPLETE_STOP:
                    self._execute_parking_complete_stop_phase()
                elif self.current_phase == ParkingPhase.FINAL_FORWARD:
                    self._execute_final_forward_phase()
                elif self.current_phase == ParkingPhase.COMPLETED:
                    self._execute_completed_phase()
                    
            except Exception as e:
                print(f"❌ 주차 실행 중 오류: {e}")
                self.stop_parking()
    
    def _execute_waiting_phase(self):
        """대기 단계 실행"""
        if not self.phase_states['initial_forward_started']:
            self._set_phase(ParkingPhase.INITIAL_FORWARD)
            self.phase_states['initial_forward_started'] = True
    
    def _execute_initial_forward_phase(self):
        """초기 전진 단계 실행"""
        self._move_forward()
        self._straight_steering()
        self.status_message = "똑바로 전진 중..."
        
        if self._check_sensor_detection():
            self._set_phase(ParkingPhase.FIRST_STOP)
    
    def _execute_first_stop_phase(self):
        """첫 번째 정지 단계 실행"""
        self._stop_vehicle()
        self.phase_states['first_stop_completed'] = True
        self.status_message = "첫 번째 정지 완료"
        self._set_phase(ParkingPhase.LEFT_TURN_FORWARD)
    
    def _execute_left_turn_forward_phase(self):
        """좌회전 전진 단계 실행"""
        if not self.phase_states['left_turn_started']:
            self._turn_left()
            self._move_forward()
            self.phase_states['left_turn_started'] = True
            self.status_message = "왼쪽 조향 전진 중..."
        
        if self._check_second_stop_condition():
            self._set_phase(ParkingPhase.SECOND_STOP)
    
    def _execute_second_stop_phase(self):
        """두 번째 정지 단계 실행"""
        self._stop_vehicle()
        self.phase_states['second_stop_completed'] = True
        self.status_message = "두 번째 정지 완료"
        self._set_phase(ParkingPhase.RIGHT_TURN_BACKWARD)
    
    def _execute_right_turn_backward_phase(self):
        """우회전 후진 단계 실행 - 조향각 점진적 조정 추가"""
        if not self.phase_states['right_turn_started']:
            self._turn_right()
            self._move_backward()
            self.phase_states['right_turn_started'] = True
            self.backward_start_time = time.time()
            self.status_message = "오른쪽 조향 후진 중..."
        
        # 조향각 점진적 조정 (시뮬레이션과 동일)
        if self.backward_start_time is not None:
            elapsed_time = time.time() - self.backward_start_time
            if elapsed_time < 2.0:
                # 2초에 걸쳐 조향각을 13도에서 0도로 줄임
                steering_reduction = (elapsed_time / 2.0) * self.parking_config['right_turn_angle']
                current_steering = max(0, self.parking_config['right_turn_angle'] - steering_reduction)
                
                # 조향각에 따른 조향 설정 - 각도 기반으로 수정
                if current_steering > 0:
                    self._set_steering_angle(current_steering)  # 현재 조향각으로 설정
                else:
                    self._straight_steering()  # 직진으로 전환
                
                self.status_message = f"조향각 점진적 조정 중... ({current_steering:.1f}도)"
            else:
                self._straight_steering()  # 2초 후 직진으로 전환
        
        if self._check_backward_completion():
            self._set_phase(ParkingPhase.STRAIGHT_BACKWARD)
    
    def _execute_straight_backward_phase(self):
        """정방향 후진 단계 실행"""
        if not self.phase_states['straight_backward_started']:
            self._straight_steering()
            self._move_backward()
            self.phase_states['straight_backward_started'] = True
            self.straight_backward_start_time = time.time()
            self.status_message = "정방향 후진 중..."
        
        if self._check_time_elapsed(self.straight_backward_start_time, 
                                  self.parking_config['straight_backward_duration']):
            self._set_phase(ParkingPhase.ALIGNMENT)
    
    def _execute_alignment_phase(self):
        """정렬 단계 실행"""
        if not self.phase_states['alignment_completed']:
            self._move_backward()
            self.status_message = "차량 정렬 중..."
        
        if self._check_alignment_completion():
            self.phase_states['alignment_completed'] = True
            self._stop_vehicle()
            self._set_phase(ParkingPhase.POSITION_CHECK)
    
    def _execute_position_check_phase(self):
        """위치 확인 단계 실행"""
        if self._check_position_correction_needed():
            self._set_phase(ParkingPhase.CORRECTION)
        else:
            self._set_phase(ParkingPhase.PARKING_COMPLETE_STOP)
    
    def _execute_correction_phase(self):
        """수정 단계 실행 - 점진적 각도 변화 추가"""
        if not self.phase_states['correction_started']:
            self.correction_start_time = time.time()
            self.phase_states['correction_started'] = True
            self._move_forward()
            
            # 수정 방향 결정 (한 번만)
            if "좌측으로 치우침" in self.status_message:
                self.correction_direction = 1  # 우회전 (양수)
                self.status_message = "우회전으로 수정 시작..."
            else:
                self.correction_direction = -1  # 좌회전 (음수)
                self.status_message = "좌회전으로 수정 시작..."
        
        # 점진적 각도 변화 적용
        if self.correction_start_time is not None:
            elapsed_time = time.time() - self.correction_start_time
            correction_duration = self.parking_config['correction_duration']  # 2.0초
            
            if elapsed_time < correction_duration:
                # 2초에 걸쳐 한쪽 끝에서 반대쪽 끝으로 점진적 이동 (시뮬레이션과 일치)
                if self.correction_direction == 1:  # 우회전 (좌측으로 치우침)
                    start_angle = self.parking_config['correction_angle']  # +15도
                    end_angle = -self.parking_config['correction_angle']   # -15도
                else:  # 좌회전 (우측으로 치우침)
                    start_angle = -self.parking_config['correction_angle']  # -15도
                    end_angle = self.parking_config['correction_angle']     # +15도
                
                progress_ratio = elapsed_time / correction_duration
                current_angle = start_angle + (end_angle - start_angle) * progress_ratio
                
                # 현재 각도로 조향 설정
                self._set_steering_angle(current_angle)
                self.status_message = f"점진적 수정 중... ({current_angle:.1f}도)"
            else:
                # 수정 완료 - 직진으로 복귀
                self._straight_steering()
                self.status_message = "수정 완료! 직진으로 복귀..."
        
        if self._check_time_elapsed(self.correction_start_time, 
                                  self.parking_config['correction_duration']):
            self.phase_states['correction_completed'] = True
            self._stop_vehicle()
            self._set_phase(ParkingPhase.POST_CORRECTION_BACKWARD)
    
    def _execute_post_correction_backward_phase(self):
        """수정 후 후진 단계 실행 - 추가 후진 시간 로직 추가"""
        if not self.phase_states['post_correction_backward_started']:
            self._straight_steering()
            self._move_backward()
            self.phase_states['post_correction_backward_started'] = True
            self.post_correction_backward_start_time = time.time()
            self.status_message = "수정 후 정방향 후진 중..."
        
        # front_right 센서 거리 확인
        front_right_distance = self._get_sensor_distance("front_right")
        
        # front_right가 40cm 이하가 되면 추가 후진 시작 시간 기록
        if front_right_distance <= self.parking_config['stop_distance']:
            if self.additional_backward_start_time is None:
                self.additional_backward_start_time = time.time()
                self.status_message = "front_right 40cm 이하! 추가 정방향 후진 시작..."
            elif self._check_time_elapsed(self.additional_backward_start_time, 
                                        self.parking_config['additional_backward_duration']):
                self._stop_vehicle()
                self.status_message = "수정 후 정방향 후진 완료!"
                self._set_phase(ParkingPhase.PARKING_COMPLETE_STOP)
    
    def _execute_parking_complete_stop_phase(self):
        """주차 완료 정지 단계 실행"""
        if not self.phase_states['parking_completion_stop_started']:
            self._stop_vehicle()
            self.phase_states['parking_completion_stop_started'] = True
            self.parking_completion_stop_start_time = time.time()
            self.status_message = "주차 완료! 2초 정지 중..."
        
        if self._check_time_elapsed(self.parking_completion_stop_start_time, 
                                  self.parking_config['parking_stop_duration']):
            self._set_phase(ParkingPhase.FINAL_FORWARD)
    
    def _execute_final_forward_phase(self):
        """최종 전진 단계 실행 - 우회전 로직 완전 구현"""
        if not self.phase_states['parking_completion_forward_started']:
            self._straight_steering()
            self._move_forward()
            self.phase_states['parking_completion_forward_started'] = True
            self.status_message = "최종 정방향 주행 중..."
        
        # rear_right 갑작스러운 증가 감지
        rear_right_current = self._get_sensor_distance("rear_right")
        if (self.previous_distances["rear_right"] > 0 and 
            rear_right_current > self.previous_distances["rear_right"] + 15):
            
            # 우회전 시작
            if not self.phase_states['right_turn_after_increase_started']:
                self.right_turn_after_increase_start_time = time.time()
                self.phase_states['right_turn_after_increase_started'] = True
                # 시뮬레이션과 일치하도록 20도로 설정
                self._set_steering_angle(self.parking_config['final_right_turn_angle'])  # 20도
                self.status_message = "오른쪽 조향 중..."
            
            # 우회전 완료 확인
            elif self._check_time_elapsed(self.right_turn_after_increase_start_time, 
                                        self.parking_config['right_turn_duration']):
                self._straight_steering()  # 직진으로 복귀
                self.status_message = "오른쪽 조향 완료! 정방향 주행 시작..."
                self._set_phase(ParkingPhase.COMPLETED)
    
    def _execute_completed_phase(self):
        """완료 단계 실행"""
        self._stop_vehicle()
        self.parking_completed = True
        self.is_parking_active = False
        self.status_message = "주차 완료!"
        print("🎉 주차 완료!")
    
    def get_status(self):
        """현재 상태 반환"""
        with self._lock:
            return {
                'phase': self.current_phase.name,
                'phase_number': self.current_phase.value,
                'status_message': self.status_message,
                'is_active': self.is_parking_active,
                'is_completed': self.parking_completed,
                'sensor_distances': self.sensor_distances.copy(),
                'sensor_flags': self.sensor_flags.copy()
            }
    
    def get_parking_config(self):
        """주차 설정 반환"""
        return self.parking_config.copy()
    
    def update_parking_config(self, new_config):
        """주차 설정 업데이트"""
        with self._lock:
            self.parking_config.update(new_config)
    
    def emergency_stop(self):
        """비상 정지"""
        with self._lock:
            self._stop_vehicle()
            self.stop_parking()
            self.status_message = "비상 정지!"
            print("🚨 비상 정지!")
    
    def reset_system(self):
        """시스템 리셋"""
        with self._lock:
            self._stop_vehicle()
            self.is_parking_active = False
            self.parking_completed = False
            self.current_phase = ParkingPhase.WAITING
            self.status_message = "시스템 리셋됨"
            self._reset_phase_states()
            
            # 센서 플래그 초기화
            for key in self.sensor_flags:
                self.sensor_flags[key] = False
            
            print("🔄 시스템 리셋 완료") 