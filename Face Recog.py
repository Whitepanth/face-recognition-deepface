# FACE RECOGNITION, GENDER, AGE, EMOTION DETECTION USING DEEPFACE

# IMPORT PACKAGES
import os
import cv2
import numpy as np
from deepface import DeepFace

# CREATE DATASET
dir = "Dataset"
os.makedirs(dir, exist_ok=True)

def create_dataset(name):
    person = os.path.join(dir, name)
    os.makedirs(person, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot Capture Image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")\
                    .detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y+h, x:x+w]
            face_path = os.path.join(person, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Capture Face In Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()

def train_dataset():
    embedding = {}
    for i in os.listdir(dir):
        person = os.path.join(dir, i)
        if os.path.isdir(person):
            embedding[i] = []
            for img_name in os.listdir(person):
                img_path = os.path.join(person, img_name)
                try:
                    rep = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
                    embedding[i].append(rep[0]["embedding"])
                except Exception as e:
                    print(f"Failed to train image {img_path}: {e}")
    return embedding

# RECOGNIZE FACE, AGE, GENDER, EMOTION
def recognize_Face(embedding):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")\
                    .detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            try:
                analyse = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)

                if isinstance(analyse, list):
                    analyse = analyse[0]

                age = analyse["age"]
                gender = analyse["gender"]
                emotion = max(analyse["emotion"], key=analyse["emotion"].get)

                face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                match = None
                max_similarity = -1

                for person_name, person_embeddings in embedding.items():
                    for embed in person_embeddings:
                        similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = person_name

                if max_similarity > 0.7:
                    label = f"{match} ({max_similarity:.2f})"
                else:
                    label = "Unknown Person"

                display_text = f"{label}, Age: {int(age)}, Gender: {gender}, Emotion: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            except Exception as e:
                print("Cannot recognize face:", e)

        cv2.imshow("Recognize Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# MAIN EXECUTION
if __name__ == "__main__":
    print("1. Create Face Dataset\n2. Train Face Dataset\n3. Recognize Faces")
    choice = input("Enter Your Choice: ")

    if choice == "1":
        name = input("Enter Name of the Person: ")
        create_dataset(name)
    elif choice == "2":
        embedding = train_dataset()
        np.save("embedding.npy", embedding)
    elif choice == "3":
        if os.path.exists("embedding.npy"):
            embedding = np.load("embedding.npy", allow_pickle=True).item()
            recognize_Face(embedding)
        else:
            print("Data Not Found")
    else:
        print("Invalid Choice")
