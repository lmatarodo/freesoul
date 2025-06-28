# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import cv2
import numpy as np
import os
import time
from pathlib import Path

class BirdViewConverter:
    def __init__(self):
        """
        비디오를 버드아이 뷰로 변환하는 클래스
        """
        self.output_dir = "birdview_frames"
        
    def create_output_directory(self):
        """출력 디렉토리 생성"""
        Path(self.output_dir).mkdir(exist_ok=True)
        print(f"출력 디렉토리 생성: {self.output_dir}")
    
    def warpping(self, image, srcmat, dstmat):
        """
        원근 변환(Perspective Transform) 수행
        :param image: 원본 이미지
        :param srcmat: 원근 변환 전 좌표
        :param dstmat: 원근 변환 후 좌표
        :return: 변환된 이미지, 역변환 행렬
        """
        h, w = image.shape[0], image.shape[1]
        transform_matrix = cv2.getPerspectiveTransform(srcmat, dstmat)
        minv = cv2.getPerspectiveTransform(dstmat, srcmat)  # 역변환용
        warped = cv2.warpPerspective(image, transform_matrix, (w, h))
        return warped, minv
    
    def bird_convert(self, img, srcmat, dstmat):
        """
        주어진 srcmat, dstmat 좌표를 이용하여 버드아이 뷰(bird's-eye view)로 변환
        :param img: 원본 이미지
        :param srcmat: 원근 변환 전 좌표
        :param dstmat: 원근 변환 후 좌표 
        :return: 변환된 이미지
        """
        srcmat = np.float32(srcmat)
        dstmat = np.float32(dstmat)
        img_warped, _ = self.warpping(img, srcmat, dstmat)
        return img_warped
    
    def roi_rectangle_below(self, img, cutting_idx):
        """
        이미지 하부를 잘라서 ROI 추출
        :param img: 원본 이미지
        :param cutting_idx: 위에서부터 얼마나 자를지(픽셀 인덱스)
        :return: 잘려진 ROI 이미지
        """
        return img[cutting_idx:, :]
    
    def process_video_to_birdview_frames(self, video_path, output_size=(256, 256), 
                                       save_original=False, save_roi=False):
        """
        비디오를 버드아이 뷰 프레임들로 변환
        :param video_path: 입력 비디오 경로
        :param output_size: 출력 이미지 크기 (width, height)
        :param save_original: 원본 프레임도 저장할지 여부
        :param save_roi: ROI 추출 후 이미지도 저장할지 여부
        """
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"비디오를 열 수 없습니다: {video_path}")
            return
        
        # 비디오 정보 가져오기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"비디오 정보:")
        print(f"  - 총 프레임 수: {total_frames}")
        print(f"  - FPS: {fps:.2f}")
        print(f"  - 해상도: {width}x{height}")
        print(f"  - 출력 크기: {output_size[0]}x{output_size[1]}")
        
        # 출력 디렉토리 생성
        self.create_output_directory()
        
        # 버드아이 뷰 변환을 위한 좌표 설정
        # 원본 이미지 크기에 맞게 조정
        dst_mat = [
            [round(width * 0.3), 0],
            [round(width * 0.7), 0],
            [round(width * 0.7), height],
            [round(width * 0.3), height]
        ]
        
        # 실제 환경에 맞게 조정된 src_mat (카메라 위치에 따라 달라짐)
        src_mat = [
            [238, 316],
            [402, 313],
            [501, 476],
            [155, 476]
        ]
        
        # 좌표가 이미지 범위를 벗어나지 않도록 조정
        src_mat = self.adjust_coordinates(src_mat, width, height)
        
        print(f"원근 변환 좌표:")
        print(f"  - src_mat: {src_mat}")
        print(f"  - dst_mat: {dst_mat}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 진행률 출력
            if frame_count % 30 == 0:  # 30프레임마다 진행률 출력
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (frame_count / total_frames)
                remaining_time = estimated_total - elapsed_time
                print(f"진행률: {progress:.1f}% ({frame_count}/{total_frames}) "
                      f"예상 남은 시간: {remaining_time:.1f}초")
            
            # 1. 버드아이 뷰 변환
            bird_img = self.bird_convert(frame, srcmat=src_mat, dstmat=dst_mat)
            
            # 2. ROI 추출 (하단 부분만)
            roi_image = self.roi_rectangle_below(bird_img, cutting_idx=300)
            
            # 3. 출력 크기에 맞게 리사이즈
            resized_img = cv2.resize(roi_image, output_size)
            
            # 4. 프레임 저장
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(self.output_dir, frame_filename)
            cv2.imwrite(frame_path, resized_img)
            
            # 추가 저장 옵션
            if save_original:
                original_filename = f"original_{frame_count:06d}.jpg"
                original_path = os.path.join(self.output_dir, original_filename)
                cv2.imwrite(original_path, frame)
            
            if save_roi:
                roi_filename = f"roi_{frame_count:06d}.jpg"
                roi_path = os.path.join(self.output_dir, roi_filename)
                cv2.imwrite(roi_path, roi_image)
        
        # 비디오 캡처 객체 해제
        cap.release()
        
        total_time = time.time() - start_time
        print(f"\n변환 완료!")
        print(f"  - 총 처리 시간: {total_time:.2f}초")
        print(f"  - 평균 처리 속도: {frame_count/total_time:.2f} FPS")
        print(f"  - 저장된 프레임 수: {frame_count}")
        print(f"  - 저장 위치: {os.path.abspath(self.output_dir)}")
    
    def adjust_coordinates(self, src_mat, width, height):
        """
        좌표가 이미지 범위를 벗어나지 않도록 조정
        """
        adjusted = []
        for x, y in src_mat:
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            adjusted.append([x, y])
        return adjusted
    
    def create_video_from_frames(self, input_dir, output_video_path, fps=30):
        """
        프레임들을 다시 비디오로 합치기
        """
        frame_files = sorted([f for f in os.listdir(input_dir) if f.startswith('frame_') and f.endswith('.jpg')])
        
        if not frame_files:
            print("프레임 파일을 찾을 수 없습니다.")
            return
        
        # 첫 번째 프레임으로 크기 확인
        first_frame_path = os.path.join(input_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, layers = first_frame.shape
        
        # 비디오 작성자 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        print(f"비디오 생성 중: {output_video_path}")
        print(f"  - 해상도: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - 총 프레임: {len(frame_files)}")
        
        for frame_file in frame_files:
            frame_path = os.path.join(input_dir, frame_file)
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        print(f"비디오 생성 완료: {output_video_path}")

def main():
    """메인 함수"""
    converter = BirdViewConverter()
    
    # 비디오 경로 설정
    video_path = "test_video.mp4"
    
    # 출력 설정
    output_size = (256, 256)  # YOLO 모델 입력 크기에 맞춤
    save_original = False     # 원본 프레임 저장 여부
    save_roi = False          # ROI 이미지 저장 여부
    
    print("=== 비디오를 버드아이 뷰 프레임으로 변환 ===")
    print(f"입력 비디오: {video_path}")
    print(f"출력 크기: {output_size[0]}x{output_size[1]}")
    
    # 비디오를 프레임으로 변환
    converter.process_video_to_birdview_frames(
        video_path=video_path,
        output_size=output_size,
        save_original=save_original,
        save_roi=save_roi
    )
    
    # 선택사항: 프레임들을 다시 비디오로 합치기
    create_video = input("\n프레임들을 다시 비디오로 합치시겠습니까? (y/n): ").lower().strip()
    if create_video == 'y':
        output_video_path = "birdview_video.mp4"
        converter.create_video_from_frames(
            input_dir=converter.output_dir,
            output_video_path=output_video_path,
            fps=30
        )

if __name__ == "__main__":
    main() 