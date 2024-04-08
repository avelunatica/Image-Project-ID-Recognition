import re
import cv2
import math
import numpy as np
from scipy import ndimage
import pytesseract
from pytesseract import Output

def rotate(im):
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 190, minLineLength=100, maxLineGap=5)
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(im, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(im, median_angle)
    
    return img_rotated

def trim_dni(img):
    THIN_THRESHOLD = 300
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 3)
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        if h < THIN_THRESHOLD or w < THIN_THRESHOLD:
            continue
        roi = img[y:y + h, x:x + w]
        cv2.rectangle(im, (x, y), (x + w, y + h), (200, 0, 0), 2)
    return roi

def number_ID(img):
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize image
    thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Noise removal and contour smoothing, morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    opening_image = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 1)

    # Reverse image
    invert_image = 255 - opening_image
    dni = "[0-9]{8}[A-Z]{1}"
    d = pytesseract.image_to_data(invert_image, output_type=Output.DICT)

    #print(d)
    n_boxes = len(d['text'])
    #print(d['text']) #print all the detected text in the image
    for i in range(n_boxes):
        #print(d['text'][i])
        if re.match(dni, d['text'][i]):
            #print(f"El DNI es: {d['text'][i]}")
            return d['text'][i]
        #else:
        #    print("El DNI no es válido")
    return None

def name_ID(img):
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize image
    thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Noise removal and contour smoothing, morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    opening_image = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Reverse image
    invert_image = 255 - opening_image

    # Define a regular expression pattern for the label "NOMBRE/NOME"
    label_pattern = r"NOMBRE"

    # Use Tesseract to extract text data
    d = pytesseract.image_to_data(invert_image, output_type=Output.DICT)

    n_boxes = len(d['text'])
    print(d['text'])

    # Iterate over the detected text
    for i in range(n_boxes):
        # Check if the text matches the label pattern
        if re.search(label_pattern, d['text'][i]):
            # If the label is found, extract the text below it as the name
            name_box_index = i + 1  # Assuming the name is in the box below the label
            if name_box_index < n_boxes:
                return d['text'][name_box_index]

    # If no name is found, return None
    return None

def dni_surname(img):
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize image
    thresh_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Noise removal and contour smoothing, morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    opening_image = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Reverse image
    invert_image = 255 - opening_image

    # Define a regular expression pattern for the label "APELLIDO"
    surname_label_pattern = r"APELLIDO"

    # Use Tesseract to extract text data
    d = pytesseract.image_to_data(invert_image, output_type=Output.DICT)

    n_boxes = len(d['text'])

    # Initialize variable for surname
    surname = None

    # Iterate over the detected text
    for i in range(n_boxes):
        # Check if the text contains the surname label pattern
        if re.search(surname_label_pattern, d['text'][i], flags=re.IGNORECASE):
            # If the label is found, extract the text below it as the surname
            surname_box_index = i + 1  # Assuming the surname is in the box below the label
            if surname_box_index < n_boxes:
                surname = d['text'][surname_box_index]

                # Check if there is a subsequent box and append its text to the surname
                next_box_index = surname_box_index + 1
                while next_box_index < n_boxes and d['text'][next_box_index].strip() != '':
                    surname += ' ' + d['text'][next_box_index]
                    next_box_index += 1

    # Return the extracted surname (may be None if not found)
    return surname

if __name__ == '__main__':
    imagen = cv2.imread("a.jpg")
    print(f"number_ID(imagen): {number_ID(imagen)}")

    #hola=rotate(imagen)
    #cv2.imshow("hola",hola)
    #cv2.waitKey(0)

    print(f"name_ID(imagen): {name_ID(imagen)}")

    print(f"dni_surname(imagen): {dni_surname(imagen)}")


