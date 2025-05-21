import pyrealsense2 as rs
import cv2
import numpy as np
import time
import requests

# Load Haar Cascade
cascPath = "Data/haarcascade_frontalface_default.xml"  # Use your custom bottle cap cascade
faceCascade = cv2.CascadeClassifier(cascPath)
apiUrl = ""

# Setup RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# Grace period tracking
entry_time = None
status = "NO OBJECT"

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect bottle caps
        caps = faceCascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        misaligned = False
        cap_count = len(caps)
        cap_found = False
        alignmentCheck = {'isMisaligned': misaligned}

        # Define and draw alignment box (inline here)
        align_box_x, align_box_y, align_box_w, align_box_h = 120, 100, 400, 280
        cv2.rectangle(color_image,
                      (align_box_x, align_box_y),
                      (align_box_x + align_box_w, align_box_y + align_box_h),
                      (255, 255, 0), 2)

        # Check detection against alignment zone
        for (x, y, w, h) in caps:
            cap_found = True
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if (x < align_box_x or y < align_box_y or
                x + w > align_box_x + align_box_w or
                y + h > align_box_y + align_box_h):
                misaligned = True
                requests.post(apiUrl, json=alignmentCheck)

        # Apply inline grace period logic
        current_time = time.time()
        grace_period = 2.0  # seconds

        if cap_found:
            if entry_time is None:
                entry_time = current_time

            if current_time - entry_time > grace_period:
                status = "MISALIGNED" if misaligned else "ALIGNED"
        else:
            status = "NO OBJECT"
            entry_time = None

        # Display status
        color = (0, 255, 0) if status == "ALIGNED" else (0, 0, 255)
        if status == "NO OBJECT":
            color = (180, 180, 180)

        cv2.putText(color_image, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow('Bottle Cap Alignment Checker', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
