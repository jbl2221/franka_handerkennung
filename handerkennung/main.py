import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import time
from concurrent.futures import ThreadPoolExecutor
import threading

def configure_realsense():
    # Initialize the pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image, depth_frame

def detect_hands(color_image, hands_processor):
    results = hands_processor.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    return results

def draw_hand_landmarks(color_image, results, mp_hands):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return color_image

def get_distance_to_hand(depth_image, hand_landmarks, depth_frame, mp_hands):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    x, y = int(wrist.x * depth_image.shape[1]), int(wrist.y * depth_image.shape[0])

    # Ensure wrist coordinates are within the depth image dimensions
    if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
        distance = depth_frame.get_distance(x, y)
        return distance
    else:
        return None

def print_distance_at_intervals(distances, interval=0.5):
    while True:
        if distances:
            if isinstance(distances[-1], float):
                print(f'Distance to wrist: {distances[-1]:.2f} meters')
            else:
                print(distances[-1])
        else:
            print('No hand detected')
        time.sleep(interval)

def main():
    pipeline = configure_realsense()
    mp_hands = mp.solutions.hands
    hands_processor = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    frame_skip = 2
    frame_count = 0
    distances = []

    # Start the printing thread
    printing_thread = threading.Thread(target=print_distance_at_intervals, args=(distances,))
    printing_thread.daemon = True
    printing_thread.start()

    try:
        while True:
            start_time = time.time()

            depth_image, color_image, depth_frame = get_frames(pipeline)
            if depth_image is None or color_image is None:
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            results = detect_hands(color_image, hands_processor)
            color_image = draw_hand_landmarks(color_image, results, mp_hands)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    distance = get_distance_to_hand(depth_image, hand_landmarks, depth_frame, mp_hands)
                    if distance is not None:
                        if len(distances) > 0:
                            distances[0] = distance
                        else:
                            distances.append(distance)
                    else:
                        if len(distances) > 0:
                            distances[0] = 'No hand detected'
                        else:
                            distances.append('No hand detected')
            else:
                if len(distances) > 0:
                    distances[0] = 'No hand detected'
                else:
                    distances.append('No hand detected')

            cv2.imshow('Hand Detection', color_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            # print(f'FPS: {fps:.2f}')

    finally:
        hands_processor.close()
        cv2.destroyAllWindows()
        pipeline.stop()

if __name__ == "__main__":
    main()
