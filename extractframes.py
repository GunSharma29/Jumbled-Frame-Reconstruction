import cv2
import os

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_folder, f"frame_{count:03d}.png")  # lossless PNG format
        cv2.imwrite(filename, frame)
        count += 1
    
    cap.release()
    print(f"Extracted {count} frames to folder '{output_folder}'.")

if __name__ == "__main__":
    extract_frames("jumbled_video.mp4", "frames")
