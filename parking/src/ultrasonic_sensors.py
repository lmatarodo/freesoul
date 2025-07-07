import time
import RPi.GPIO as GPIO

class UltrasonicSensors:
    def __init__(self):
        # GPIO 핀 설정 (BCM 모드)
        self.sensors = [
            {"trigger": 17, "echo": 18, "name": "front_right"},
            {"trigger": 22, "echo": 23, "name": "middle_left"},
            {"trigger": 24, "echo": 25, "name": "middle_right"},
            {"trigger": 26, "echo": 27, "name": "rear_left"},
            {"trigger": 28, "echo": 29, "name": "rear_right"}
        ]
        
        # GPIO 설정
        GPIO.setmode(GPIO.BCM)
        for sensor in self.sensors:
            GPIO.setup(sensor["trigger"], GPIO.OUT)
            GPIO.setup(sensor["echo"], GPIO.IN)
            GPIO.output(sensor["trigger"], GPIO.LOW)
    
    def read_distance(self, trigger_pin, echo_pin):
        """단일 센서 거리 측정"""
        try:
            # 트리거 신호
            GPIO.output(trigger_pin, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(trigger_pin, GPIO.LOW)
            
            # 에코 신호 대기
            start_time = time.time()
            timeout = start_time + 0.1
            
            while GPIO.input(echo_pin) == GPIO.LOW and time.time() < timeout:
                start_time = time.time()
            
            while GPIO.input(echo_pin) == GPIO.HIGH and time.time() < timeout:
                stop_time = time.time()
            
            # 거리 계산 (cm)
            elapsed_time = stop_time - start_time
            distance = (elapsed_time * 34300) / 2
            
            if 2 <= distance <= 400:
                return distance
            else:
                return -1
                
        except:
            return -1
    
    def read_all_sensors(self):
        """모든 센서 읽기"""
        distances = {}
        for sensor in self.sensors:
            distance = self.read_distance(sensor["trigger"], sensor["echo"])
            distances[sensor["name"]] = distance
        return distances
    
    def cleanup(self):
        """GPIO 정리"""
        GPIO.cleanup()

# 사용 예시
if __name__ == "__main__":
    sensors = UltrasonicSensors()
    
    try:
        print("초음파 센서 테스트 시작...")
        print("Ctrl+C로 종료")
        
        while True:
            distances = sensors.read_all_sensors()
            print("\n" + "="*50)
            for name, distance in distances.items():
                if distance > 0:
                    print(f"{name}: {distance:.1f}cm")
                else:
                    print(f"{name}: 측정 실패")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n프로그램 종료")
    finally:
        sensors.cleanup() 