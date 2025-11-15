import cv2
import mediapipe as mp
import time
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def get_finger_states(hand_landmarks):
    fingers = []
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    is_right_hand = wrist.x < middle_mcp.x

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    if is_right_hand:
        thumb_extended = thumb_tip.x > thumb_ip.x + 0.02  
    else:
        thumb_extended = thumb_tip.x < thumb_ip.x - 0.02 
    fingers.append(1 if thumb_extended else 0)

    finger_data = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP)
    ]

    for tip_idx, pip_idx, mcp_idx in finger_data:
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        mcp = hand_landmarks.landmark[mcp_idx]

        extended = tip.y < pip.y - 0.03 and tip.y < mcp.y - 0.02
        fingers.append(1 if extended else 0)

    return fingers

def recognize_pose(fingers):
    if fingers == [1, 0, 0, 0, 0]: 
        return "thumbs_up"
    elif fingers == [0, 1, 1, 0, 0]: 
        return "peace_sign"
    elif fingers == [0, 0, 0, 0, 0]:  
        return "fist"
    elif fingers == [0, 1, 1, 1, 1] or fingers == [0, 1, 1, 1, 0] or fingers == [0, 1, 1, 0, 1]:  
        return "open_hand"
    else:
        return "unknown"

def create_confetti(frame_width, frame_height):
    particles = []
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for _ in range(50):  
        x = random.randint(0, frame_width)
        y = random.randint(0, frame_height // 2)  
        color = random.choice(colors)
        speed = random.uniform(2, 5)
        particles.append([x, y, color, speed])
    return particles

def update_confetti(particles, frame_height):
    for particle in particles:
        particle[1] += particle[3]  
        if particle[1] > frame_height:
            particle[1] = 0  
            particle[0] = random.randint(0, 900)  
    return particles

def draw_confetti(frame, particles):
    for particle in particles:
        x, y, color, _ = particle
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open webcam")
        return

    score = 0
    current_pose = None
    pose_start_time = None
    time_limit = 10 # seconds  
    feedback = ""
    feedback_time = 0
    pose_change_delay = 1.0  # seconds

    confetti_particles = [] 
    effect_duration = 0

    delay_until = 0  
    poses = {
        "Thumbs Up": "thumbs_up",
        "Peace Sign": "peace_sign",
        "Fist": "fist",
        "Open Hand": "open_hand"
    }

    print("hand pose match")
    print("press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        detected_pose = "unknown"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = get_finger_states(hand_landmarks)
                detected_pose = recognize_pose(fingers)

        current_time = time.time()

        if current_time < delay_until:
            cv2.imshow('hand pose match', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if current_pose is None:
            current_pose = random.choice(list(poses.keys()))
            pose_start_time = current_time
            feedback = ""
        elif current_time - pose_start_time > time_limit:
            feedback = f"time up {current_pose}"
            feedback_time = current_time
            cv2.putText(frame, "X", (frame.shape[1]//2 - 50, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            delay_until = current_time + 1  # second delay
            current_pose = None
        elif detected_pose == poses[current_pose] and current_time - (pose_start_time or 0) > 0.5:  # Shorter delay to avoid instant detection
            score += 1
            feedback = f"new score: {score}"
            feedback_time = current_time
            confetti_particles = create_confetti(frame.shape[1], frame.shape[0])
            effect_duration = current_time + 1  # second of confetti
            delay_until = current_time + 1  # second delay
            current_pose = None

        if current_time < effect_duration and confetti_particles:
            confetti_particles = update_confetti(confetti_particles, frame.shape[0])
            frame = draw_confetti(frame, confetti_particles)
            cv2.waitKey(10)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cv2.putText(frame, f"make pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"make pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if pose_start_time:
            time_left = max(0, int(time_limit - (current_time - pose_start_time)))
            cv2.putText(frame, f"time: {time_left}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"time: {time_left}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"score: {score}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"score: {score}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"detected: {detected_pose}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"detected: {detected_pose}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if feedback and current_time - feedback_time < 3:  
            cv2.putText(frame, feedback, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, feedback, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('hand pose match', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"final score: {score}")

if __name__ == "__main__":
    main()
