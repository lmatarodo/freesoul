# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import cv2
import numpy as np
import os
import time

def convert_video_to_birdview_frames(video_path, output_dir="birdview_frames", 
                                   output_size=(256, 256), start_frame=0, end_frame=None):
    """
    비디오를 버드아이 뷰 프레임들로 변환하는 간단한 함수
    
    Args:
        video_path: 입력 비디오 경로
        output_dir: 출력 디렉토리
        output_size: 출력 이미지 크기 (width, height)
        start_frame: 시작 프레임 번호 (0부터 시작)
        end_frame: 끝 프레임 번호 (None이면 전체)
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리: {output_dir}")
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"비디오 정보:")
    print(f"  - 총 프레임: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - 해상도: {width}x{height}")
    
    # 처리할 프레임 범위 설정
    if end_frame is None:
        end_frame = total_frames
    
    # 버드아이 뷰 변환 좌표 설정
    # 원본 이미지 크기에 맞게 조정
    dst_mat = np.float32([
        [width * 0.3, 0],      # 좌상단
        [width * 0.7, 0],      # 우상단
        [width * 0.7, height], # 우하단
        [width * 0.3, height]  # 좌하단
    ])
    
    # 실제 환경에 맞게 조정된 src_mat (카메라 위치에 따라 달라짐)
    src_mat = np.float32([
        [250, 316],  # 좌상단
        [380, 316],  # 우상단
        [450, 476],  # 우하단
        [200, 476]   # 좌하단
    ])
    
    # 좌표가 이미지 범위를 벗어나지 않도록 조정
    src_mat[:, 0] = np.clip(src_mat[:, 0], 0, width - 1)
    src_mat[:, 1] = np.clip(src_mat[:, 1], 0, height - 1)
    
    print(f"원근 변환 좌표:")
    print(f"  - src_mat: {src_mat.tolist()}")
    print(f"  - dst_mat: {dst_mat.tolist()}")
    
    # 변환 행렬 계산
    transform_matrix = cv2.getPerspectiveTransform(src_mat, dst_mat)
    
    # 시작 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    processed_count = 0
    start_time = time.time()
    
    print(f"\n변환 시작 (프레임 {start_frame} ~ {end_frame})...")
    
    save_interval = int(fps * 0.5)  # 0.5초 간격 (예: 30fps면 15)
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 0.5초 간격으로만 저장
        if (frame_count - start_frame) % save_interval == 0:
            # 1. 버드아이 뷰 변환
            bird_img = cv2.warpPerspective(frame, transform_matrix, (width, height))
            
            # 2. ROI 추출 (하단 부분만)
            roi_image = bird_img[300:, :]  # 300픽셀부터 아래쪽만
            
            # 3. 출력 크기에 맞게 리사이즈
            resized_img = cv2.resize(roi_image, output_size)
            
            # 4. 프레임 저장
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, resized_img)
        
        processed_count += 1
        
        # 진행률 출력 (10프레임마다)
        if processed_count % 10 == 0:
            progress = ((frame_count - start_frame + 1) / (end_frame - start_frame)) * 100
            elapsed_time = time.time() - start_time
            if processed_count > 0:
                estimated_total = elapsed_time / (processed_count / (end_frame - start_frame))
                remaining_time = estimated_total - elapsed_time
                print(f"진행률: {progress:.1f}% ({frame_count}/{end_frame}) "
                      f"남은 시간: {remaining_time:.1f}초")
        
        frame_count += 1
    
    cap.release()
    
    total_time = time.time() - start_time
    print(f"\n변환 완료!")
    print(f"  - 처리된 프레임: {processed_count}")
    print(f"  - 총 처리 시간: {total_time:.2f}초")
    print(f"  - 평균 처리 속도: {processed_count/total_time:.2f} FPS")
    print(f"  - 저장 위치: {os.path.abspath(output_dir)}")

def create_video_from_frames(input_dir, output_video_path, fps=30):
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

if __name__ == "__main__":
    # 설정
    video_path = "test_video.mp4"
    output_dir = "birdview_frames"
    output_size = (256, 256)  # YOLO 모델 입력 크기
    
    # 전체 비디오 변환
    print("=== test_video를 버드아이 뷰 프레임으로 변환 ===")
    convert_video_to_birdview_frames(
        video_path=video_path,
        output_dir=output_dir,
        output_size=output_size
    )
    
    # 선택사항: 특정 구간만 변환하고 싶다면 아래 주석을 해제
    # convert_video_to_birdview_frames(
    #     video_path=video_path,
    #     output_dir=output_dir + "_partial",
    #     output_size=output_size,
    #     start_frame=100,  # 100번째 프레임부터
    #     end_frame=200     # 200번째 프레임까지
    # )
    
    # 선택사항: 프레임들을 다시 비디오로 합치기
    create_video_choice = input("\n프레임들을 다시 비디오로 합치시겠습니까? (y/n): ").lower().strip()
    if create_video_choice == 'y':
        create_video_from_frames(output_dir, "birdview_video.mp4", fps=30) 