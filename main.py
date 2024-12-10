
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from face_model import recognize_face, load_model

clf, reverse_label_map = load_model()
people = list(reverse_label_map.values()) + ["Unknown"]
confidences = [0.0]
# ทดสอบด้วยรูปใหม่
test_image_path = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/img"
for person_path in os.listdir(test_image_path):
    test_image = cv2.imread(person_path)
    if test_image is None:
        print(f"Error: Image not found at {test_image_path}")
    
    test_image = cv2.resize(test_image, (160, 160))

    name, confidence = recognize_face(test_image, clf, reverse_label_map)
    print(f"Recognized: {name} with confidence {confidence:.2f}")

    # Retrieve index when name is in reverse_label_map
    if name in reverse_label_map.values():
        reverse_label_map.values()
    else:
        people["Unknown"] 

# Save the bar chart as an image file without displaying it
plt.figure(figsize=(10, 6))
plt.bar(categories, confidence_scores, color='skyblue', alpha=0.7)
plt.ylim(0, 1)
plt.xlabel("Categories", fontsize=14)
plt.ylabel("Confidence Scores", fontsize=14)
plt.title("Confidence Scores by Categories", fontsize=16)
plt.axhline(y=0.6, color='red', linestyle='--', label='Threshold (0.6)')
plt.legend()

plt.tight_layout()
plt.savefig("C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/models/confidence_scores_chart.png")

# Load the pre-trained model and config file
model_path = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/premodels/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "C:/Users/sahathat.y/source/repos/face_recognition_v3/myenv/premodels/deploy.prototxt"

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
# ax.legend(loc="best")

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
