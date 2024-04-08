# Importing necessary libraries
import cv2
import face_recognition
import numpy as np

# Function to detect faces in an image and compare them
def detectar_caras_y_comparar(image_path, path):
    # Import cascade file for facial recognition
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Load an image for testing
    test_image = cv2.imread(image_path)
    
    # Resize the image if its dimensions are larger than 500 pixels
    if max(test_image.shape[:2]) > 500:
        test_image = cv2.resize(test_image, (0, 0), fx=500/max(test_image.shape[:2]), fy=500/max(test_image.shape[:2]))

    # Convert the test image to grayscale
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascades
    faces = face_cascade.detectMultiScale(test_image_gray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbors

    # Check if there are exactly two detected faces
    if len(faces) == 2:
        # Initialize variables to store the detected face encodings
        detected_face_encodings = []
        face_saved = []

        # Iterate through detected faces
        for idx, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around the face
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Crop the face
            face_roi = test_image[y:y + h, x:x + w]
            
            # Resize the face image to improve face recognition accuracy
            face_roi_resized = cv2.resize(face_roi, (150, 150))
            face_saved.append(face_roi_resized)

            # Show the resized face image for debugging
            cv2.imshow(f'Resized Face {idx+1}', face_roi_resized)
            cv2.waitKey(0)

            # Perform face recognition
            face_encoding = face_recognition.face_encodings(face_roi_resized)

            # Check if a face is recognized
            if face_encoding:
                # Add the detected face encoding to the list
                detected_face_encodings.append(face_encoding[0])

                # Save the face image when two faces are detected
                if len(detected_face_encodings) == 2:
                    if np.linalg.norm(detected_face_encodings[0] - detected_face_encodings[1]) < 0.6:
                        cv2.imwrite(f"{path}/cara1igual.jpg", face_saved[0])
                        cv2.imwrite(f"{path}/cara2igual.jpg", face_saved[1])
                    else:
                        cv2.imwrite(f"{path}/cara1diferente.jpg", face_saved[0])
                        cv2.imwrite(f"{path}/cara2diferente.jpg", face_saved[1])

        # Check if there are exactly two detected face encodings
        if len(detected_face_encodings) == 2:
            # Compare the two detected face encodings
            distance = np.linalg.norm(detected_face_encodings[0] - detected_face_encodings[1])

            # Set a threshold for face similarity
            if distance < 0.6:
                return "Cara 1 y Cara 2 son iguales."  # Faces are the same
            else:
                return "Cara 1 y Cara 2 son diferentes."  # Faces are different
        else:
            return "No se detecta correctamente la foto del DNI."  # Faces not detected correctly
    else:
        return "No se pudieron reconocer caras en la imagen."  # Faces not recognized in the image

# Example of usage
#PATH OF THE IMAGE AND WHERE YOU WANT TO SAVE THE CROPPED FACES
image_path = "C:/Users/Usuario/Desktop/train_data/Entrega/Prueba No Iguales.jpg"
path = "C:/Users/Usuario/Desktop/train_data/Entrega/Datos DNi"
resultado = detectar_caras_y_comparar(image_path, path)
print(resultado)

