import pytesseract
import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk


# Configurar la ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def ocr(img):
    # # Leer la imagen, convertir a escala de grises y aplicar umbralización binaria inversa con Otsu
    # _, thresh_img = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # # Definir el kernel, aplicar operación morfológica de apertura e invertir la imagen
    # img = 255 - cv.morphologyEx(thresh_img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)), iterations=1)

    # # Empequeñecer la imagen si es necesario
    # if max(img.shape[:2]) > 1000:
    #     img = cv.resize(img, (0, 0), fx=500/max(img.shape[:2]), fy=500/max(img.shape[:2]))
    
    image=img

    cv.imshow('DNi Information', img)
    cv.waitKey(0)

    # Usa Tesseract para realizar OCR en la imagen procesada utilizando ESPAÑOL
    ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6 -l spa')
    # Devolver el texto reconocido
    return ocr_text

def yolo_to_absolute(height,width, box):

    x_center, y_center, box_width, box_height = box
    # Calcular coordenadas absolutas
    x1 = int((x_center - box_width/2) * width)
    y1 = int((y_center - box_height/2) * height)
    x2 = int((x_center + box_width/2) * width)
    y2 = int((y_center + box_height/2) * height)
    return y1, y2, x1, x2


def choose_and_return_index(image_paths, title):
    def on_image_click(image_number):
        root.destroy()  # Cerrar la ventana cuando se hace clic en una imagen
        choose_and_return_index.selected_index = image_number

    # Crear la ventana principal
    root = tk.Tk()
    root.title(title)

    # Añadir un título en grande
    title_label = tk.Label(root, text=title, font=("Helvetica", 16))
    title_label.pack()

    # Cargar las tres imágenes de ejemplo
    images = [Image.open(path) for path in image_paths]
    image_tks = [ImageTk.PhotoImage(image) for image in images]

    # Crear etiquetas para mostrar las imágenes
    labels = [tk.Label(root, image=image_tk) for image_tk in image_tks]

    # Colocar las etiquetas en la ventana
    for i, label in enumerate(labels):
        label.bind("<Button-1>", lambda event, i=i: on_image_click(i + 1))  # "+1" para que sea 1-indexado
        label.pack()

    # Iniciar el bucle principal
    root.mainloop()

    # Devolver el número de imagen que se hizo clic
    return choose_and_return_index.selected_index


