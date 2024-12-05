import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from collections import deque
import matplotlib.pyplot as plt

# โหลดโมเดล FaceNet
mtcnn = MTCNN(keep_all=True)
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

def get_face_embeddingMTCNN(face):
    """Create face embedding from the image"""
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Use MTCNN to detect faces in the image
    faces = mtcnn(face_rgb)
    
    if faces is None:
        return None  # No faces detected
    
    # Assuming we only care about the first detected face
    face_embedding = None
    for detected_face in faces:
        with torch.no_grad():
            embedding = model(detected_face.unsqueeze(0)).cpu().numpy().flatten()
        face_embedding = embedding # Return embedding and confidence for the first face
    
    return face_embedding

def get_face_embeddingMTCNNgraph(face):
    """Create face embedding from the image"""
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Use MTCNN to detect faces in the image
    boxes, probs = mtcnn.detect(face_rgb)  # Get both bounding boxes and probabilities (confidence)
    
    if boxes is None:
        return None, 0  # No faces detected
    
    # Assuming we only care about the first detected face
    face_embedding = None
    max_confidence = 0  # To store maximum confidence value for the detected faces

    for i, box in enumerate(boxes):
        confidence = probs[i]  # Confidence of the i-th detected face
        max_confidence = max(max_confidence, confidence)  # Keep track of the highest confidence

        # Extract face region of interest (ROI)
        detected_face = face_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        # Check if the face size is valid (avoid very small faces)
        if detected_face.shape[0] < 20 or detected_face.shape[1] < 20:
            print(f"Detected face is too small: {detected_face.shape}")
            continue  # Skip this face if it's too small
        
        # Resize the face to 160x160 pixels (required by the model)
        detected_face_resized = cv2.resize(detected_face, (160, 160))

        # Extract embedding from face ROI
        detected_face_tensor = torch.tensor(np.array(detected_face_resized) / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            embedding = model(detected_face_tensor).cpu().numpy().flatten()
        
        face_embedding = embedding  # You can store embeddings for each face if needed

    return face_embedding, max_confidence

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
faces_folder = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/faces"
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
    embedding = get_face_embeddingMTCNN(face)
    label = clf.predict([embedding])[0]
    confidence = clf.predict_proba([embedding]).max()
    if confidence < threshold:
        return "Unknown", confidence
    name = reverse_label_map[label]
    return name, confidence

# ทดสอบด้วยรูปใหม่
test_image_path = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/img/benz.jpeg"
test_image = cv2.imread(test_image_path)
if test_image is None:
    print(f"Error: Image not found at {test_image_path}")
test_image = cv2.resize(test_image, (160, 160))

name, confidence = recognize_face(test_image, clf, reverse_label_map)
print(f"Recognized: {name} with confidence {confidence:.2f}")

# Load the pre-trained model and config file
model_path = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/models/deploy.prototxt"

# ใช้ OpenCV DNN สำหรับตรวจจับใบหน้า
face_net = cv2.dnn.readNetFromCaffe(config_path, model_path)
# Initialize variables for real-time graph plotting
confidence_history = deque(maxlen=30)  # To store the last 30 confidence values

cap = cv2.VideoCapture(0)  # 0 หมายถึงกล้องหลัก

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess for DNN
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#     face_net.setInput(blob)
#     detections = face_net.forward()

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.6:  # Threshold for face detection
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # Extract face ROI
#             face = frame[startY:endY, startX:endX]
#             if face.size == 0:  # ตรวจสอบใบหน้าไม่ว่าง
#                 continue

#             try:
#                 # Predict face name and confidence
#                 name, conf = recognize_face(face, clf, reverse_label_map)

#                 # Draw bounding box and label
#                 text = f"{name}: {conf:.2f}"
#                 cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#                 cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             except Exception as e:
#                 print(f"Error processing face: {e}")

#     # Show the video feed
#     cv2.imshow("Real-Time Face Recognition", frame)

#     # Break the loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# Initialize the matplotlib plot
fig, ax = plt.subplots(figsize=(8, 6))  # Make sure the plot has a suitable size
ax.set_ylim(0, 1)  # Confidence values range between 0 and 1
ax.set_title("Real-time Confidence of Face Recognition")
ax.set_xlabel("Frame")
ax.set_ylabel("Confidence")
ax.legend(loc="best")

# Real-time video stream processing
confidence_history = []
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
        if confidence > 0.6:  # Threshold for face detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:  # ตรวจสอบใบหน้าไม่ว่าง
                continue

            try:
                # Predict face name and confidence
                name, conf = recognize_face(face, clf, reverse_label_map)
                
                # If a face is detected, add confidence to the history
                confidence_history.append(conf)

                # Draw bounding box and label
                text = f"{name}: {conf:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")

    # Clear the previous graph
    ax.clear()

    threshold = 0.6

    # Plot the updated confidence values
    ax.plot(confidence_history, label="Confidence")
    ax.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    
    ax.set_ylim(0, 1)  # Confidence values are between 0 and 1
    ax.set_title("Real-time Confidence of Face Recognition")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Confidence")
    ax.legend(loc="best")

    # Draw the updated graph on the screen
    plt.draw()  # Force redrawing the plot
    
    # Draw the updated graph on the screen
    plt.pause(0.01)

    # Show the video frame with confidence overlay
    cv2.imshow("Face Recognition with Confidence", frame)
    
    if len(confidence_history) > 100:  # Keep the last 100 frames
        confidence_history.pop(0)

    # Exit the video stream on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.close()
