import cv2
import mediapipe as mp

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Chuyển ảnh sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Xử lý bằng MediaPipe để nhận diện bàn tay
    results = hands.process(frame_rgb)

    # Vẽ các landmarks nếu có nhận diện bàn tay
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Tính toán số lượng ngón tay (ví dụ dựa trên vị trí các landmark)
            # Sử dụng các điểm landmarks để xác định nếu các ngón tay đang mở hay không
            finger_count = 0

            # Kiểm tra ngón cái
            if landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
                finger_count += 1

            # Kiểm tra các ngón còn lại
            for i in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]:
                if landmarks.landmark[i].y < landmarks.landmark[i - 2].y:
                    finger_count += 1

            # Hiển thị số ngón tay
            cv2.putText(frame, f'Ngon tay: {finger_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow('Finger Count', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
