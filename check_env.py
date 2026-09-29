import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

print("torch:", torch.__version__)
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
print("CUDA available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)  # downloads weights on first run
print("Models loaded")

# Webcam test: press q to quit
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        print("Webcam not found")
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV uses BGR, MTCNN expects RGB
    boxes, _ = mtcnn.detect(rgb)
    if boxes is not None:
        for (x1, y1, x2, y2) in boxes.astype(int):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Face detection test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()