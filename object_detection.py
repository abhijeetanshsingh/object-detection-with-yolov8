import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # make sure you are connected to internet for first time
cap = cv2.VideoCapture(0)  # 0 is default webcam, replace with 'video.mp4' for a video file
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
# Setup video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))

init_once = False
trackers = cv2.MultiTracker_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not init_once:
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
            tracker = cv2.TrackerCSRT_create()
            trackers.add(tracker, frame, (x1, y1, x2 - x1, y2 - y1))

        init_once = True

    success, tracked_objects = trackers.update(frame)

    for i, newbox in enumerate(tracked_objects):
        x, y, w, h = [int(v) for v in newbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("YOLOv8 Object Detection + Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

# 🆕 Paste full tracking + video saving code here
...
