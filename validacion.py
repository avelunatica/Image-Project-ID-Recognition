# Importing necessary libraries
import cv2
import face_recognition
import numpy as np

# Function to detect faces in two images and compare them
def validacion(image_path_1, image_path_2, path):
    # Import cascade file for facial recognition
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Load images for testing
    test_image_1 = image_path_1
    test_image_2 = image_path_2
    
    # Resize the images if their dimensions are larger than 500 pixels
    for idx, test_image in enumerate([test_image_1, test_image_2]):
        if max(test_image.shape[:2]) > 500:
            test_image = cv2.resize(test_image, (0, 0), fx=500/max(test_image.shape[:2]), fy=500/max(test_image.shape[:2]))

        # Convert the test image to grayscale
        test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascades
        faces = face_cascade.detectMultiScale(test_image_gray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbors

        # Check if there is exactly one detected face
        if len(faces) == 1:
            # Initialize variables to store the detected face encoding
            detected_face_encoding = None

            # Draw rectangle around the face
            x, y, w, h = faces[0]
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Crop the face
            face_roi = test_image[y:y + h, x:x + w]

            # Resize the face image to improve face recognition accuracy
            face_roi_resized = cv2.resize(face_roi, (150, 150))

            # Show the resized face image for debugging
            cv2.imshow(f'Resized Face {idx+1}', face_roi_resized)
            cv2.waitKey(0)

            # Perform face recognition
            face_encoding = face_recognition.face_encodings(face_roi_resized)

            # Check if a face is recognized
            if face_encoding:
                # Add the detected face encoding to the variable
                detected_face_encoding = face_encoding[0]

            # Save the face image
            cv2.imwrite(f"{path}/cara{idx+1}.jpg", face_roi_resized)

            if detected_face_encoding is not None:
                # Save the detected face encoding
                np.save(f"{path}/cara{idx+1}_encoding.npy", detected_face_encoding)

        else:
            return 3  # Exactly one face should be detected in each image

    # Load the saved face encodings
    detected_face_encoding_1 = np.load(f"{path}/cara1_encoding.npy")
    detected_face_encoding_2 = np.load(f"{path}/cara2_encoding.npy")

    # Compare the two detected face encodings
    distance = np.linalg.norm(detected_face_encoding_1 - detected_face_encoding_2)

    # Set a threshold for face similarity
    print(f"distance: {distance}")
    if distance < 0.5:
        return 0  # Faces are the same
    else:
        return 1  # Faces are different