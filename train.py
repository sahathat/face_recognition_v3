import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from face_model import load_embeddings, save_model

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

save_model(clf, reverse_label_map)