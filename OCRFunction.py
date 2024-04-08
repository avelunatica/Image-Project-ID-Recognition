import pytesseract
import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk


# Configurar la ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr(img):
    # Leer la imagen, convertir a escala de grises y aplicar umbralización binaria inversa con Otsu
    _, thresh_img = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Definir el kernel, aplicar operación morfológica de apertura e invertir la imagen
    img = 255 - cv.morphologyEx(thresh_img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)), iterations=1)

    # Empequeñecer la imagen si es necesario
    if max(img.shape[:2]) > 1000:
        img = cv.resize(img, (0, 0), fx=500/max(img.shape[:2]), fy=500/max(img.shape[:2]))
    

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


def choose_and_return_index(image_paths):
    def on_image_click(image_number):
        root.destroy()  # Cerrar la ventana cuando se hace clic en una imagen
        choose_and_return_index.selected_index = image_number

    # Crear la ventana principal
    root = tk.Tk()
    root.title("What type of DNi is it?")

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



#USE OF CODE

# Ruta a la imagen que deseas procesar CAMBIAAAAAR
#path= 'C:/Users/Usuario/Desktop/P3 AIMV/images/DNItypes/'
img_path='C:/Users/Usuario/Desktop/Captura de pantalla 2023-12-26 231320.png'
#img_path = path + 'Front3-0.jpg'
#img_path= path + 'DNi Ultimo.jpg'
#img_path = path + 'DNi Azul.jpg'
DNiSide=input("Is the introduced image the front or back part of the ID card? (F/B)")

if DNiSide.lower()=='b':
    crop=cv.imread(img_path)
    y1, y2, x1, x2 = 544, 740, 50, 1230
    # Realizar el recorte
    MRZ = crop[y1:y2, x1:x2]
    cv.imshow("MRZ",MRZ)
    MRZ_text=ocr(MRZ)
    print(MRZ_text)

elif DNiSide.lower()=='f':
    #TIPOS DE DNIS. PONER RUTAS DE DONDE SE GUARDARON
    image_paths = ["C:/Users/Usuario/Desktop/P3 AIMV/DNi Types/First DNi.jpeg",
                   "C:/Users/Usuario/Desktop/P3 AIMV/DNi Types/3.0 DNi.jpg",
                   "C:/Users/Usuario/Desktop/P3 AIMV/DNi Types/Latest DNi.jpg"]

    selected_index = choose_and_return_index(image_paths)
    # Imprimir el índice seleccionado
    if selected_index==1:
        print(f"DNi azul")
        crop=cv.imread(img_path)
        height, width, channels = crop.shape
        yolo_coordinates =(0.130694, 0.946204, 0.229841, 0.072284)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        DNiNumber = crop[y1:y2, x1:x2]
        cv.imshow("DNi",DNiNumber)
        cv.waitKey(0)
        DNiNumber_text=ocr(DNiNumber)
        print(DNiNumber_text)
        #Second Surname1
        yolo_coordinates =(0.421306, 0.180642, 0.276251, 0.042393)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Surname1 = crop[y1:y2, x1:x2]
        cv.imshow("Surnames",Surname1)
        cv.waitKey(0)
        Surname1_text=ocr(Surname1)
        print(Surname1_text)
        #Surname2
        yolo_coordinates =(0.421817, 0.259672, 0.275229, 0.044624)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Surname2 = crop[y1:y2, x1:x2]
        cv.imshow("Surname",Surname2)
        cv.waitKey(0)
        Surname2_text=ocr(Surname2)
        print(Surname2_text)
        #Third Name
        yolo_coordinates =(0.421817, 0.333301, 0.279092, 0.046111)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Name = crop[y1:y2, x1:x2]
        cv.imshow("Name",Name)
        cv.waitKey(0)
        Name_text=ocr(Name)
        print(Name_text)
        #Signature
        yolo_coordinates =(0.452905, 0.822185, 0.357798, 0.319358)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Signature = crop[y1:y2, x1:x2]
        cv.imshow("Signature",Signature)
        cv.waitKey(0)

    elif selected_index==2:
        print(f"DNi 3.0")
        crop=cv.imread(img_path)
        height, width, channels = crop.shape
        yolo_coordinates = (0.225965, 0.920127, 0.290578, 0.107949)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        DNiNumber = crop[y1:y2, x1:x2]
        cv.imshow("DNi",DNiNumber)
        cv.waitKey(0)
        DNiNumber_text=ocr(DNiNumber)
        print(DNiNumber_text)
        #Second Surnames
        yolo_coordinates =(0.562761, 0.342397, 0.343592, 0.079081)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Surnames = crop[y1:y2, x1:x2]
        cv.imshow("Surnames",Surnames)
        cv.waitKey(0)
        Surnames_text=ocr(Surnames)
        print(Surnames_text)
        #Third Name
        yolo_coordinates =(0.568008, 0.456028, 0.350223, 0.034550)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Name = crop[y1:y2, x1:x2]
        cv.imshow("Name",Name)
        cv.waitKey(0)
        Name_text=ocr(Name)
        print(Name_text)
        #Signature
        yolo_coordinates =(0.571309, 0.882894, 0.395329, 0.2280991)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Signature = crop[y1:y2, x1:x2]
        cv.imshow("Signature",Signature)
        cv.waitKey(0)


    elif selected_index==3:
        print(f"Ultimo DNi")
        crop=cv.imread(img_path)
        height, width, channels = crop.shape
        yolo_coordinates = (0.583486, 0.203377, 0.282569, 0.078262)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        DNiNumber = crop[y1:y2, x1:x2]
        cv.imshow("DNi",DNiNumber)
        cv.waitKey(0)
        DNiNumber_text=ocr(DNiNumber)
        print(DNiNumber_text)
        #Second Surnames
        yolo_coordinates =(0.555535, 0.341060, 0.340001, 0.086958)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Surnames = crop[y1:y2, x1:x2]
        cv.imshow("Surnames",Surnames)
        cv.waitKey(0)
        Surnames_text=ocr(Surnames)
        print(Surnames_text)
        #Third Name
        yolo_coordinates =(0.553638, 0.453029, 0.336070, 0.039665)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Name = crop[y1:y2, x1:x2]
        cv.imshow("Name",Name)
        cv.waitKey(0)
        Name_text=ocr(Name)
        print(Name_text)
        #Signature
        yolo_coordinates =(0.577982, 0.828024, 0.392661, 0.162321)
        y1,y2,x1,x2 = yolo_to_absolute(height,width, yolo_coordinates)
        Signature = crop[y1:y2, x1:x2]
        cv.imshow("Signature",Signature)
        cv.waitKey(0)

