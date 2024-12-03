import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image

# โหลดโมเดล FaceNet
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(face):
    """สร้าง face embedding จากภาพ"""
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face)
    face_pil = face_pil.resize((160, 160))
    face_tensor = torch.tensor(np.array(face_pil) / 255.0).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        embedding = model(face_tensor).numpy().flatten()
    return embedding

def load_embeddings(folder_path):
    """โหลด embeddings และ labels จากโฟลเดอร์"""
    embeddings = []
    labels = []
    label_map = {}  # เก็บ mapping ของชื่อเป็น label ID
    current_label = 0

    for person_folder in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_path):
            if person_folder not in label_map:
                label_map[person_folder] = current_label
                current_label += 1

            for file in os.listdir(person_path):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(person_path, file)
                    image = cv2.imread(image_path)
                    try:
                        # Resize และสร้าง embedding
                        image = cv2.resize(image, (160, 160))
                        embedding = get_face_embedding(image)
                        embeddings.append(embedding)
                        labels.append(label_map[person_folder])
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

    return np.array(embeddings), np.array(labels), label_map

# โหลดข้อมูล
faces_folder = "F:/pythonProject/face_recognition/myenv/faces"
X, y, label_map = load_embeddings(faces_folder)

# แบ่งชุดข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# ประเมินผล
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Mapping กลับจาก label เป็นชื่อคน
reverse_label_map = {v: k for k, v in label_map.items()}


def recognize_face(face, clf, reverse_label_map, threshold=0.6):
    """ทำนายชื่อของใบหน้าใหม่"""
    embedding = get_face_embedding(face)
    label = clf.predict([embedding])[0]
    confidence = clf.predict_proba([embedding]).max()
    if confidence < threshold:
        return "Unknown", confidence
    name = reverse_label_map[label]
    return name, confidence

# # ทดสอบด้วยรูปใหม่
# test_image_path = "F:/pythonProject/face_recognition/myenv/img/hirun.jpeg"
# test_image = cv2.imread(test_image_path)
# if test_image is None:
#     print(f"Error: Image not found at {test_image_path}")
# test_image = cv2.resize(test_image, (160, 160))

# name, confidence = recognize_face(test_image, clf, reverse_label_map)
# print(f"Recognized: {name} with confidence {confidence:.2f}")

# Load the pre-trained model and config file
model_path = "F:/pythonProject/face_recognition/myenv/models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "F:/pythonProject/face_recognition/myenv/models/deploy.prototxt"

# ใช้ OpenCV DNN สำหรับตรวจจับใบหน้า
face_net = cv2.dnn.readNetFromCaffe(config_path, model_path)

cap = cv2.VideoCapture(0)  # 0 หมายถึงกล้องหลัก

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess for DNN
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Threshold for face detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:  # ตรวจสอบใบหน้าไม่ว่าง
                continue

            try:
                # Predict face name and confidence
                name, conf = recognize_face(face, clf, reverse_label_map)

                # Draw bounding box and label
                text = f"{name}: {conf:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")

    # Show the video feed
    cv2.imshow("Real-Time Face Recognition", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
