import pyrealsense2 as rs
import cv2
import numpy as np
import time
import requests
import threading

# Load Haar Cascade
cascPath = "Data/cascade.xml"
capCascade = cv2.CascadeClassifier(cascPath)
apiUrl = "http://10.56.203.96:5000/crate/"

def send_alignment_update(data):
    if not apiUrl:
        return
    try:
        response = requests.put(apiUrl, json=data)
        print(f"API Response: {response.status_code}")
        print(data)
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")

# Setup RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

entry_time = None
status = "NO OBJECT"
last_status = None
last_cap_count = -1
last_api_call_time = 0
api_call_interval = 2

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
        caps = capCascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(50, 50))

        cap_count = len(caps)
        cap_found = cap_count > 0
        misaligned = False

        # Alignment box
        align_box_x, align_box_y, align_box_w, align_box_h = 120, 100, 400, 280
        cv2.rectangle(color_image,
                      (align_box_x, align_box_y),
                      (align_box_x + align_box_w, align_box_y + align_box_h),
                      (255, 255, 0), 2)

        for (x, y, w, h) in caps:
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if (x < align_box_x or y < align_box_y or
                x + w > align_box_x + align_box_w or
                y + h > align_box_y + align_box_h):
                misaligned = True

        current_time = time.time()
        grace_period = 2.0

        if cap_found:
            if entry_time is None:
                entry_time = current_time

            if current_time - entry_time > grace_period:
                status = "MISALIGNED" if misaligned else "ALIGNED"
        else:
            status = "NO OBJECT"
            entry_time = None

        # API call if status or cap count changed, or enough time has passed
        if (status != last_status or cap_count != last_cap_count or
                current_time - last_api_call_time > api_call_interval):
            last_status = status
            last_cap_count = cap_count
            last_api_call_time = current_time

            if cap_found:
                alignmentCheck = {
                    "alignment": str(misaligned),
                    "caps": int(cap_count)
                }
                threading.Thread(target=send_alignment_update, args=(alignmentCheck,), daemon=True).start()

        # Display status on image
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