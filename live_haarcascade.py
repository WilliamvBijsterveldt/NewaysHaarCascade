import pyrealsense2 as rs
import cv2
import sys
import time
import numpy as np

cascPath = "haarcascade_frontalface_alt.xml"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start streaming
pipeline.start(config)

# Align depth to color stream
align_to = rs.stream.color
align = rs.align(align_to)

faceCascade = cv2.CascadeClassifier(cascPath)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale for detection
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect beer bottle caps
        faces = faceCascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected objects
        for (x, y, w, h) in faces:
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the output
        cv2.imshow('Face Detection', color_image)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
