# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import time
from threading import Lock
import spidev
import keyboard
import numpy as np

class MotorController:
    def __init__(self, motors):
        # 기본 모터 설정
        self.motors = motors
        self.size = 600600  # 2ms
        self._left_speed = 0
        self._right_speed = 0
        self._steering_speed = 0
        self.steering_angle = 0
        
        # 조향 관련 변수
        self.current_duty = self.size // 2  # 현재 duty 값 (50%)
        self.min_duty = self.size // 2      # 최소 duty 값 (50%)
        self.max_duty = int(self.size * 0.8)  # 최대 duty 값 (80%)
        self.duty_step = int(self.size * 0.02)  # duty 증가량 (2%)
        self.last_steering_time = time.time()
        
        # 제어 변수
        # self.auto_duty = self.min_duty
        self.manual_steering_angle = 0
        self.manual_speed = 0
        
        # SPI 설정
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 20000000
        self.spi.mode = 0b00
        
        # 저항 값 범위 설정
        self.resistance_most_left = 3300 
        self.resistance_most_right = 2500
    @property
    def steering_speed(self):
        return self._steering_speed

    @steering_speed.setter
    def steering_speed(self, value):
        self._steering_speed = value
        self.control_motors(self._steering_speed)  # 속도 변경 시 자동으로 반영
        
    @property
    def left_speed(self):
        return self._left_speed

    @left_speed.setter
    def left_speed(self, value):
        self._left_speed = value
        self.set_left_speed(self._left_speed)  # 속도 변경 시 자동으로 반영

    @property
    def right_speed(self):
        return self._right_speed

    @right_speed.setter
    def right_speed(self, value):
        self._right_speed = value
        self.set_right_speed(self._right_speed)  # 속도 변경 시 자동으로 반영

    def init_motors(self):
        """모터 초기화"""
        for name, motor in self.motors.items():
            motor.write(0x00, self.size)     # size
            motor.write(0x04, self.min_duty)  # 초기 duty 50%
            motor.write(0x08, 0)             # valid

    def reset_motor_values(self):
        """모터 값 안전 초기화"""
        self.left_speed = 0
        self.right_speed = 0
        self.steering_speed = 0
        self.steering_angle = 0
        self.manual_speed = 0
        self.manual_steering_angle = 0
        self.current_duty = self.min_duty
        
        # 모든 모터 정지
        for motor in self.motors.values():
            motor.write(0x08, 0)
        
        # duty 값 초기화
        for motor in self.motors.values():
            motor.write(0x04, self.min_duty)

    def right(self, steering_speed, control_mode=1):
        """우회전 제어"""
        print(f"[MOTOR_DEBUG] 우회전 제어: steering_speed={steering_speed}, control_mode={control_mode}")
        
        if control_mode == 1:  # 자율주행 모드
            duty_percent = abs(steering_speed) / 100
            duty = int(self.size * duty_percent)
            print(f"[MOTOR_DEBUG] 자율주행 모드: duty_percent={duty_percent:.2f}, duty={duty}")
        else:  # 수동 주행 모드
            current_time = time.time()
            if current_time - self.last_steering_time > 0.05:
                self.current_duty = min(self.max_duty, self.current_duty + self.duty_step)
                self.last_steering_time = current_time
            duty = self.current_duty
            print(f"[MOTOR_DEBUG] 수동 주행 모드: duty={duty}")
            
        print(f"[MOTOR_DEBUG] motor_4(좌회전) 비활성화, motor_5(우회전) 활성화")
        self.motors['motor_4'].write(0x08, 0)  # valid  steering_left
        self.motors['motor_5'].write(0x08, 1)  # valid  steering_right
        self.motors['motor_5'].write(0x04, duty)

    def left(self, steering_speed, control_mode=1):
        """좌회전 제어"""
        print(f"[MOTOR_DEBUG] 좌회전 제어: steering_speed={steering_speed}, control_mode={control_mode}")
        
        if control_mode == 1:  # 자율주행 모드
            duty_percent = abs(steering_speed) / 100
            duty = int(self.size * duty_percent)
            print(f"[MOTOR_DEBUG] 자율주행 모드: duty_percent={duty_percent:.2f}, duty={duty}")
        else:  # 수동 주행 모드
            current_time = time.time()
            if current_time - self.last_steering_time > 0.05:
                self.current_duty = min(self.max_duty, self.current_duty + self.duty_step)
                self.last_steering_time = current_time
            duty = self.current_duty
            print(f"[MOTOR_DEBUG] 수동 주행 모드: duty={duty}")
            
        print(f"[MOTOR_DEBUG] motor_5(우회전) 비활성화, motor_4(좌회전) 활성화")
        self.motors['motor_5'].write(0x08, 0)  # valid  steering_right
        self.motors['motor_4'].write(0x08, 1)  # valid  steering_left
        self.motors['motor_4'].write(0x04, duty)

    def stay(self, steering_speed, control_mode=1):
        """중립 상태 유지"""
        print(f"[MOTOR_DEBUG] 중립 상태 유지: steering_speed={steering_speed}, control_mode={control_mode}")
        
        if control_mode == 1:  # 자율주행 모드
            duty_percent = abs(steering_speed) / 100
            duty = int(self.size * duty_percent)
            print(f"[MOTOR_DEBUG] 자율주행 모드: duty_percent={duty_percent:.2f}, duty={duty}")
        else:  # 수동 주행 모드
            self.current_duty = self.min_duty
            duty = self.current_duty
            print(f"[MOTOR_DEBUG] 수동 주행 모드: duty={duty}")
            
        print(f"[MOTOR_DEBUG] 양쪽 모터 모두 비활성화")
        self.motors['motor_5'].write(0x08, 0)  # valid  steering_right
        self.motors['motor_4'].write(0x08, 0)  # valid  steering_left
        self.motors['motor_5'].write(0x04, duty)
        self.motors['motor_4'].write(0x04, duty)

    def set_left_speed(self, speed):
        """왼쪽 모터 속도 설정"""
        duty_percent = abs(speed) / 100
        duty = int(self.size * duty_percent)
        
        self.motors['motor_0'].write(0x04, duty)
        self.motors['motor_1'].write(0x04, duty)
        
        if speed > 0:
            self.motors['motor_0'].write(0x08, 0)
            self.motors['motor_1'].write(0x08, 1)
        else:
            self.motors['motor_0'].write(0x08, 1)
            self.motors['motor_1'].write(0x08, 0)

    def set_right_speed(self, speed):
        """오른쪽 모터 속도 설정"""
        duty_percent = abs(speed) / 100
        duty = int(self.size * duty_percent)
        
        self.motors['motor_3'].write(0x04, duty)
        self.motors['motor_2'].write(0x04, duty)
        
        if speed > 0:
            self.motors['motor_3'].write(0x08, 0)
            self.motors['motor_2'].write(0x08, 1)
        else:
            self.motors['motor_3'].write(0x08, 1)
            self.motors['motor_2'].write(0x08, 0)

    def read_adc(self):
        """ADC 값 읽기"""
        adc_response = self.spi.xfer2([0x00, 0x00])
        adc_value = ((adc_response[0] & 0x0F) << 8) | adc_response[1]
        return adc_value 

    def map_value(self, x, in_min, in_max, out_min, out_max):
        """
        x를 in_min~in_max 범위에서 out_min~out_max 범위로 매핑
        """
        if x <= in_min:
            return out_max
        elif x >= in_max:
            return out_min
        else:
            # in_min과 in_max 사이일 경우, x가 커질수록 결과가 선형적으로 감소하도록 계산
            return (in_max - x) * (out_max - out_min) / (in_max - in_min) + out_min

    def map_angle_to_range(self, angle):
        """각도를 모터 제어 범위로 매핑 (조향각 크기에 비례)"""
        # 조향각의 크기에 비례하여 -7 ~ +7 범위로 매핑
        # 30도 = 7, 0도 = 0, -30도 = -7
        max_angle = 30.0
        mapped_value = (angle / max_angle) * 7.0
        mapped_value = max(-7.0, min(7.0, mapped_value))  # -7 ~ +7 범위 제한
        return mapped_value

    def control_motors(self, angle=None, control_mode=1):
        """모터 전체 제어"""
        # 디버깅 정보 추가
        print(f"[MOTOR_DEBUG] control_motors 호출: angle={angle}, control_mode={control_mode}")
        
        mapped_resistance = self.map_value(
            self.read_adc(),
            self.resistance_most_right,
            self.resistance_most_left,
            -7, 7
        )
        
        if angle is not None:
            target_angle = self.map_angle_to_range(angle)
            # 조향각의 크기에 비례한 steering_speed 계산
            angle_magnitude = abs(angle)
            max_angle = 30.0
            proportional_speed = (angle_magnitude / max_angle) * self.steering_speed
            proportional_speed = max(10, min(self.steering_speed, proportional_speed))  # 최소 10, 최대 steering_speed
            print(f"[MOTOR_DEBUG] 입력 조향각: {angle:.2f}° → 목표 범위: {target_angle}")
            print(f"[MOTOR_DEBUG] 조향각 크기: {angle_magnitude:.2f}° → 비례 속도: {proportional_speed:.1f}")
        else:
            target_angle = self.steering_angle
            proportional_speed = self.steering_speed
            print(f"[MOTOR_DEBUG] 기본 조향각 사용: {target_angle}")
            
        print(f"[MOTOR_DEBUG] 현재 ADC 값: {self.read_adc()}, 매핑된 저항: {mapped_resistance:.2f}")
        print(f"[MOTOR_DEBUG] 목표 각도: {target_angle}, 현재 저항: {mapped_resistance:.2f}")
        
        tolerance = 0.5
        if abs(mapped_resistance - target_angle) <= tolerance:
            print(f"[MOTOR_DEBUG] 중립 상태 유지 (차이: {abs(mapped_resistance - target_angle):.2f})")
            self.stay(proportional_speed, control_mode)
        elif mapped_resistance > target_angle:
            print(f"[MOTOR_DEBUG] 좌회전 (차이: {mapped_resistance - target_angle:.2f})")
            self.left(proportional_speed, control_mode)
        else:
            print(f"[MOTOR_DEBUG] 우회전 (차이: {target_angle - mapped_resistance:.2f})")
            self.right(proportional_speed, control_mode)

    def handle_manual_control(self):
        """수동 주행 모드에서의 키보드 입력 처리"""
        if keyboard.is_pressed('w'):
            self.left_speed = min(self.left_speed + 1, 100)
            self.right_speed = min(self.right_speed + 1, 100)
            
        if keyboard.is_pressed('s'):
            self.left_speed = max(self.left_speed - 1, -100)
            self.right_speed = max(self.right_speed - 1, -100)
            
        if keyboard.is_pressed('a'):
            self.steering_angle = min(self.steering_angle - 1, 20)
            
        if keyboard.is_pressed('d'):
            self.steering_angle = max(self.steering_angle + 1, -20)
            
        if keyboard.is_pressed('r'):
            self.left_speed = 0
            self.right_speed = 0
            self.steering_angle = 0

        # 모터 제어 적용
        self.set_left_speed(self.left_speed)
        self.set_right_speed(self.right_speed)
        self.control_motors(control_mode=2)