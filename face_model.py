import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
import cv2
import joblib

def get_model():
    # โหลดโมเดล InceptionResnetV1 จาก VGGFace2
    model = InceptionResnetV1(pretrained='vggface2')

    # Freeze layers ก่อนหน้า
    for param in model.parameters():
        param.requires_grad = False

    folder_path = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/faces"
    num_classes = len(os.listdir(folder_path))
    # ปรับเปลี่ยน layer สุดท้ายเพื่อให้เหมาะกับจำนวนคลาส
    model.logits = nn.Linear(model.logits.in_features, num_classes)
    return model.eval()

def get_mtcnn():
    return MTCNN(keep_all=True)

def get_face_embedding(face):
    # โหลดโมเดล FaceNet
    model = get_model()

    """สร้าง face embedding จากภาพ"""
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face)
    face_pil = face_pil.resize((160, 160))
    face_tensor = torch.tensor(np.array(face_pil) / 255.0).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        embedding = model(face_tensor).numpy().flatten()
    return embedding

def get_face_embeddingMTCNN(face):
    # โหลดโมเดล FaceNet
    mtcnn = get_mtcnn()
    model = get_model()

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
    # โหลดโมเดล FaceNet
    mtcnn = get_mtcnn()
    model = get_model()

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
                        # Convert to RGB before passing to MTCNN
                        face_embedding = get_face_embeddingMTCNN(image)

                        if face_embedding is not None:
                            embeddings.append(face_embedding)
                            labels.append(label_map[person_folder])

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

    return np.array(embeddings), np.array(labels), label_map

def save_model(clf, reverse_label_map, model_filename="face_recognition_model.pkl", label_map_filename="label_map.pkl"):
    """Save the trained classifier and label map into the myenv/models directory."""
    
    # Define the path to save the models
    models_dir = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/models"
    
    # Ensure the directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save the classifier (SVM)
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(clf, model_path)
    print(f"Classifier saved to {model_path}")

    # Save the label map
    label_map_path = os.path.join(models_dir, label_map_filename)
    joblib.dump(reverse_label_map, label_map_path)
    print(f"Label map saved to {label_map_path}")

def load_model(model_filename="face_recognition_model.pkl", label_map_filename="label_map.pkl"):
    """Load the trained classifier and label map from the myenv/models directory."""
    
    # Define the path to the models directory
    models_dir = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/models"
    
    # Load the classifier (SVM)
    model_path = os.path.join(models_dir, model_filename)
    clf = joblib.load(model_path)
    print(f"Classifier loaded from {model_path}")

    # Load the label map
    label_map_path = os.path.join(models_dir, label_map_filename)
    reverse_label_map = joblib.load(label_map_path)
    print(f"Label map loaded from {label_map_path}")

    return clf, reverse_label_map

def recognize_face(face, clf, reverse_label_map, threshold=0.6):
    """ทำนายชื่อของใบหน้าใหม่"""
    embedding = get_face_embeddingMTCNN(face)
    label = clf.predict([embedding])[0]
    confidence = clf.predict_proba([embedding]).max()
    if confidence < threshold:
        return "Unknown", confidence
    name = reverse_label_map[label]
    return name, confidence