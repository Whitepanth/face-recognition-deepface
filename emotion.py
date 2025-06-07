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
                face_embedding = face_embedding / np.linalg.norm(face_embedding)  # Normalize the embedding

                match = None
                max_similarity = -1

                for person_name, person_embeddings in embedding.items():
                    for embed in person_embeddings:
                        embed = embed / np.linalg.norm(embed)  # Normalize the stored embedding
                        similarity = np.dot(face_embedding, embed)
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
