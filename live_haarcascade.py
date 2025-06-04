import pyrealsense2 as rs
import cv2
import numpy as np
import time
import requests
import threading
from collections import Counter


def send_alignment_update(data, api_url):
    """Send alignment and color data to API"""
    if not api_url:
        return
    try:
        response = requests.put(api_url, json=data)
        print(f"API Response: {response.status_code}")
        print(data)
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")


def evaluate_object_colour(bgr_colour, colour_ranges):
    """Evaluate if object color matches approved ranges"""
    for colour_info in colour_ranges:
        goal = colour_info["bgr"]
        tolerance = colour_info["tolerance"]
        if all(abs(c - g) <= tolerance for c, g in zip(bgr_colour, goal)):
            return f"Approved ({colour_info['colour']})"
    return "Rejected"


def count_evaluated_objects(object_evaluations):
    """Count approved, rejected, and total objects"""
    passed = sum(1 for evaluation in object_evaluations if evaluation.startswith("Approved"))
    failed = sum(1 for evaluation in object_evaluations if evaluation == "Rejected")
    total = len(object_evaluations)
    return total, passed, failed


def count_colours(object_evaluations):
    """Count objects by color"""
    colours = []
    for evaluation in object_evaluations:
        if evaluation.startswith("Approved"):
            colour = evaluation.split("(")[-1].strip(")")
            colours.append(colour)
    return Counter(colours)


def find_crate_bounds(depth_frame, color_frame):
    """Find the bounds of the crate using depth information"""
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Convert depth to millimeters and filter out invalid readings
    depth_mm = depth_image.astype(np.float32)

    # Define depth range for crate detection (adjust these values based on your setup)
    min_depth = 300  # 30cm
    max_depth = 1500  # 150cm

    # Create mask for objects in depth range
    depth_mask = np.logical_and(depth_mm > min_depth, depth_mm < max_depth)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    depth_mask = depth_mask.astype(np.uint8) * 255
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
    depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest contour (assumed to be the crate)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add some padding and ensure minimum size
    padding = 20
    min_size = 100

    x = max(0, x - padding)
    y = max(0, y - padding)
    w = max(min_size, w + 2 * padding)
    h = max(min_size, h + 2 * padding)

    # Ensure bounds don't exceed image dimensions
    img_height, img_width = depth_image.shape
    w = min(w, img_width - x)
    h = min(h, img_height - y)

    return {"x": x, "y": y, "w": w, "h": h}


def main():
    # Configuration
    cascade_path = "Data/cascade.xml"
    api_url = "http://10.56.203.96:5000/crate/"

    colour_ranges = [
        {"colour": "blue", "bgr": (70, 50, 40), "tolerance": 20},
        {"colour": "green", "bgr": (120, 120, 90), "tolerance": 20},
        {"colour": "red", "bgr": (60, 60, 200), "tolerance": 20},
    ]

    # Load cascade classifier
    cap_cascade = cv2.CascadeClassifier(cascade_path)

    # Setup RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting RealSense stream: {e}")
        return

    align_to = rs.stream.color
    align = rs.align(align_to)

    # State tracking variables
    entry_time = None
    last_status = None
    last_cap_count = -1
    last_api_call_time = 0
    api_call_interval = 2
    grace_period = 2.0

    # Crate detection variables
    fixed_align_box = None
    crate_detection_frames = 0
    crate_detection_threshold = 30  # Number of frames to establish crate position
    calibration_complete = False

    print("Press 'q' to quit the program.")
    print("Press 'c' to recalibrate crate position.")

    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Handle crate detection and calibration
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Reset calibration
                fixed_align_box = None
                crate_detection_frames = 0
                calibration_complete = False
                print("Recalibrating crate position...")

            # Detect crate bounds if not calibrated
            if not calibration_complete:
                current_crate_bounds = find_crate_bounds(depth_frame, color_frame)

                if current_crate_bounds:
                    if fixed_align_box is None:
                        fixed_align_box = current_crate_bounds
                        crate_detection_frames = 1
                    else:
                        # Check if current detection is consistent with previous
                        x_diff = abs(current_crate_bounds["x"] - fixed_align_box["x"])
                        y_diff = abs(current_crate_bounds["y"] - fixed_align_box["y"])

                        if x_diff < 30 and y_diff < 30:  # Tolerance for consistent detection
                            crate_detection_frames += 1
                            # Update with running average for stability
                            alpha = 0.1
                            fixed_align_box["x"] = int(
                                alpha * current_crate_bounds["x"] + (1 - alpha) * fixed_align_box["x"])
                            fixed_align_box["y"] = int(
                                alpha * current_crate_bounds["y"] + (1 - alpha) * fixed_align_box["y"])
                            fixed_align_box["w"] = int(
                                alpha * current_crate_bounds["w"] + (1 - alpha) * fixed_align_box["w"])
                            fixed_align_box["h"] = int(
                                alpha * current_crate_bounds["h"] + (1 - alpha) * fixed_align_box["h"])
                        else:
                            # Reset if detection is inconsistent
                            fixed_align_box = current_crate_bounds
                            crate_detection_frames = 1

                if crate_detection_frames >= crate_detection_threshold:
                    calibration_complete = True
                    print(f"Crate calibration complete! Box: {fixed_align_box}")

                # Show calibration status
                cv2.putText(color_image, f"Calibrating... {crate_detection_frames}/{crate_detection_threshold}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw alignment box
            if fixed_align_box:
                box_color = (0, 255, 0) if calibration_complete else (0, 255, 255)
                cv2.rectangle(color_image,
                              (fixed_align_box["x"], fixed_align_box["y"]),
                              (fixed_align_box["x"] + fixed_align_box["w"],
                               fixed_align_box["y"] + fixed_align_box["h"]),
                              box_color, 2)

                # Add label
                label = "Fixed Alignment Box" if calibration_complete else "Calibrating..."
                cv2.putText(color_image, label,
                            (fixed_align_box["x"], fixed_align_box["y"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Detect objects only if calibration is complete
            caps = []
            if calibration_complete:
                caps = cap_cascade.detectMultiScale(
                    gray_image,
                    scaleFactor=1.09,
                    minNeighbors=20,
                    minSize=(35, 35),
                    maxSize=(55, 55)
                )

            cap_count = len(caps)
            cap_found = cap_count > 0
            misaligned = False
            evaluations = []

            # Process each detected object
            for idx, (x, y, w, h) in enumerate(caps, start=1):
                # Color evaluation
                roi = color_image[y:y + h, x:x + w]
                colour_average = cv2.mean(roi)[:3]
                colour_average_int = tuple(int(c) for c in colour_average)
                evaluation = evaluate_object_colour(colour_average_int, colour_ranges)
                evaluations.append(evaluation)

                # Alignment check against fixed box
                if fixed_align_box:
                    if (x < fixed_align_box["x"] or y < fixed_align_box["y"] or
                            x + w > fixed_align_box["x"] + fixed_align_box["w"] or
                            y + h > fixed_align_box["y"] + fixed_align_box["h"]):
                        misaligned = True

                # Draw rectangle (green for approved, red for rejected)
                colour_rectangle = (0, 255, 0) if evaluation.startswith("Approved") else (0, 0, 255)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), colour_rectangle, 2)

                # Add labels
                cv2.putText(color_image, f"{idx}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(color_image, evaluation, (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour_rectangle, 2)
                cv2.putText(color_image, f"BGR: {colour_average_int}", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Determine alignment status
            current_time = time.time()

            if not calibration_complete:
                status = "CALIBRATING"
            elif cap_found:
                if entry_time is None:
                    entry_time = current_time

                if current_time - entry_time > grace_period:
                    status = "MISALIGNED" if misaligned else "ALIGNED"
                else:
                    status = "DETECTING..."
            else:
                status = "NO OBJECT"
                entry_time = None

            # API call logic
            if (status != last_status or cap_count != last_cap_count or
                    current_time - last_api_call_time > api_call_interval):

                last_status = status
                last_cap_count = cap_count
                last_api_call_time = current_time

                if cap_found and evaluations and calibration_complete:
                    total, passed, failed = count_evaluated_objects(evaluations)
                    colour_count = count_colours(evaluations)

                    alignment_data = {
                        "alignment": str(misaligned),
                        "caps": cap_count,
                        "total_objects": total,
                        "approved_objects": passed,
                        "rejected_objects": failed,
                        "color_breakdown": dict(colour_count),
                        "calibrated": calibration_complete
                    }
                    threading.Thread(target=send_alignment_update,
                                     args=(alignment_data, api_url), daemon=True).start()

            # Display status
            status_color = (0, 255, 0) if status == "ALIGNED" else (0, 0, 255)
            if status == "NO OBJECT":
                status_color = (180, 180, 180)
            elif status == "DETECTING...":
                status_color = (0, 255, 255)
            elif status == "CALIBRATING":
                status_color = (255, 0, 255)

            cv2.putText(color_image, status, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

            # Display color statistics
            if evaluations:
                total, passed, failed = count_evaluated_objects(evaluations)
                colour_count = count_colours(evaluations)

                info_text = f"Total: {total}, Approved: {passed}, Rejected: {failed}"
                cv2.putText(color_image, info_text, (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                y_offset = 140
                for colour, count in colour_count.items():
                    colour_text = f"{colour}: {count}"
                    cv2.putText(color_image, colour_text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25

            cv2.imshow('RealSense Object Detection & Alignment', color_image)

            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()