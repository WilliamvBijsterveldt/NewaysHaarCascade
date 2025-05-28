import pyrealsense2 as rs
import cv2
import numpy as np
from collections import Counter

def detect_objects(image, cascade_path):
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(
        gray,
        scaleFactor=1.09,
        minNeighbors=5,
        minSize=(58, 58),
        maxSize=(100, 100),
    )
    return objects

def evaluate_object_colour(bgr_colour, colour_ranges):
    for colour_info in colour_ranges:
        goal = colour_info["bgr"]
        tolerance = colour_info["tolerance"]
        if all(abs(c - g) <= tolerance for c, g in zip(bgr_colour, goal)):
            return f"Approved ({colour_info['colour']})"
    return "Rejected"

def count_evaluated_objects(object_evaluation):
    passed = sum(1 for b in object_evaluation if b.startswith("Approved"))
    failed = sum(1 for b in object_evaluation if b == "Rejected")
    total = len(object_evaluation)
    return total, passed, failed

def count_colours(object_evaluations):
    colours = []
    for evaluation in object_evaluations:
        if evaluation.startswith("Approved"):
            colour = evaluation.split("(")[-1].strip(")")
            colours.append(colour)
    return Counter(colours)

def main():
    cascade_pad = 'Data/cascade.xml'

    colour_ranges = [
        {"colour": "blue", "bgr": (70, 50, 40), "tolerance": 20},
        {"colour": "green", "bgr": (120, 120, 90), "tolerance": 20},
        {"colour": "red", "bgr": (60, 60, 200), "tolerance": 20},
    ]

    # RealSense-pipeline setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error found while starting RealSense-stream: {e}")
        return

    print("Press 'q' to quit the program.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            evaluations = []
            objects = detect_objects(frame, cascade_pad)

            for idx, (x, y, w, h) in enumerate(objects, start=1):
                roi = frame[y:y+h, x:x+w]
                colour_average = cv2.mean(roi)[:3]
                colour_average_int = tuple(int(c) for c in colour_average)

                evaluation = evaluate_object_colour(colour_average_int, colour_ranges)
                evaluations.append(evaluation)

                colour_rectangle = (0, 255, 0) if evaluation.startswith("Approved") else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour_rectangle, 2)

                cv2.putText(frame, f"{idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, evaluation, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour_rectangle, 2)
                colour_text = f"BGR: {colour_average_int}"
                cv2.putText(frame, colour_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            total, passed, failed = count_evaluated_objects(evaluations)
            colour_count = count_colours(evaluations)

            # Overlay total info
            info_text = f"Total: {total}, Approved: {passed}, Rejected: {failed}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            y_offset = 60
            for colour, total in colour_count.items():
                colour_text = f"{colour}: {total}"
                cv2.putText(frame, colour_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 25

            cv2.imshow("RealSense Object detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
