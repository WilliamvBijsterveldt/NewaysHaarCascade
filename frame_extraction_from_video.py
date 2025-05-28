import cv2
import os

# === CONFIGURATION ===
video_path = 'Data/2025-05-26 11-14-16.mkv'         # Path to your video file
output_folder = 'frames_output'        # Folder to save the frames
frame_skip = 15                         # Save 1 frame per 0.5 sec

# === CREATE OUTPUT FOLDER ===
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# === LOAD VIDEO ===
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        filename = os.path.join(output_folder, f'frame_{saved_count:05d}.jpg')
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Done! Saved {saved_count} frames to '{output_folder}'.")
