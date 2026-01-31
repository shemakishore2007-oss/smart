#Smart Safety Alert System-Hackthon prototype
import cv2
from ultralytics import YOLO
import winsound

# ==============================
# LOAD MODEL
# ==============================
model = YOLO("yolov8n.pt")

# ==============================
# PHONE CAMERA URL (CHANGE IP)
# ==============================
camera_url = "http://192.168.1.136:8080/video"   # <-- YOUR PHONE IP
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("❌ Phone camera not connected")
    exit()

print("✅ Phone camera connected")

# ==============================
# ALERT CONTROL
# ==============================
beep_count = 0
MAX_BEEPS = 20
person_in_danger_prev = False

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    h, w, _ = frame.shape

    # ==============================
    # DEFINE DANGER ZONE (BOTTOM AREA)
    # ==============================
    danger_y = int(h * 0.65)

    cv2.rectangle(frame, (0, danger_y), (w, h), (0, 0, 255), 2)
    cv2.putText(frame, "DANGER ZONE",
                (10, danger_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)

    # ==============================
    # YOLO DETECTION (NO AUTO DRAW)
    # ==============================
    results = model(frame, stream=True, verbose=False)
    person_in_danger_now = False

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])

            # PERSON CLASS
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cy = int((y1 + y2) / 2)

                # ==============================
                # RISK CHECK
                # ==============================
                if cy > danger_y:
                    person_in_danger_now = True

                    # 🔴 RISK ALERT
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "⚠ RISK ALERT",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                else:
                    # 🟢 NORMAL PERSON
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

    # ==============================
    # SOUND LOGIC (ONLY 20 TIMES)
    # ==============================
    if person_in_danger_now and not person_in_danger_prev:
        beep_count = 0

    if person_in_danger_now and beep_count < MAX_BEEPS:
        winsound.Beep(1000, 300)
        beep_count += 1

    if not person_in_danger_now:
        beep_count = 0

    person_in_danger_prev = person_in_danger_now

    # ==============================
    # DISPLAY
    # ==============================
    cv2.imshow("Smart Safety – Phone Camera Prototype", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()