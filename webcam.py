import os
import cv2
import sys
import numpy as np
import torch
import DatosFuncion
import validacion
from siftImage import match_sift_points
#Autor @RebecaGonzalez

# GLOBAL VARIABLES
morph_size = 5
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8) 
fm_value=180
fm_face_value=50
confidence_value=0.75
    
# Function to calculate the variance of Laplacian
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def crop_image(image, xmin, ymin, xmax, ymax):
    # Convert the coordinates to integers
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    # Crop the image using the defined region
    cropped_image = image[ymin:ymax, xmin:xmax]
    
    return cropped_image

# Function to detect the foreground in a given frame using a background model.
def detectForeground(frame, bgModel_RGB):
    """
    Detects the foreground in a given frame using a background model.

    Parameters:
    - frame: A numpy array representing the input frame.
    - bgModel: A background model to be applied on the frame.

    Returns:
    - Mask: A numpy array representing the foreground mask.
    """

    # Apply the background model to the frame to extract the foreground
    Mask = bgModel_RGB.apply(frame)

    # Perform noise removal
    try:
        Mask = cv2.erode(Mask, kernel)
        Mask = cv2.dilate(Mask, kernel)
    except Exception as e:
        print(f"Error on erode/dilate 1: {e}")

    # Fill holes in the mask
    morph_size = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_size + 1, 2 * morph_size + 1),
                                        (morph_size, morph_size))
    try:
        Mask = cv2.dilate(Mask, element)
        Mask = cv2.erode(Mask, element)
    except Exception as e:
        print(f"Error on erode/dilate 2: {e}")

    return Mask

def calculeMRZ(crop):
    # Calcule MRZ
    yolo_coordinates = (0.505413, 0.800350, 0.989174, 0.364141)
    height, width, _ = crop.shape
    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
    # Realizar el recorte
    MRZ = crop[y1:y2, x1:x2]

    if MRZ is not None and MRZ.shape[0] > 0 and MRZ.shape[1] > 0:
        cv2.imshow("MRZ",MRZ)
        MRZ_text=DatosFuncion.ocr(MRZ)
        #print(f"MRZ text: {MRZ_text}")
        return(MRZ_text)
    else:
        return None


def main():
    run_second_loop = False
    print("\n\033[32m╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮"
        "\n│    🪪   DNI DETECTOR AND FACE RECOGNITION 🪪     │"
        "\n╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯\033[0m\n")
    path = os.path.abspath(os.getcwd())
    # READ THE VIDEO STREAM FROM WEBCAM OR VIDEO IF IT IS PROVIDED IN CONSOLE
    console_params = sys.argv
    if len(console_params) < 2:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.abspath(os.path.join('Video','model','dnis.pt')), force_reload=True, trust_repo=True)
        model.eval()
        cap = cv2.VideoCapture(0)  # try to open webcam, this will attempt to open the first one found in the system
    elif len(console_params) == 2:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.abspath(os.path.join('Video','model','dnis.pt')), force_reload=True, trust_repo=True)
        model.eval()
        cap = cv2.VideoCapture(console_params[1])  # try to open string, this will attempt to open it as a video file
    else:
        sys.exit(" You need to provide only a video file or nothing for using a webcam")

    fps = cap.get(cv2.CAP_PROP_FPS)  # get the frames per second of the video/webcam
    #print(f"\033[1;32mFrames per second: \033[0m{fps}")

    cv2.namedWindow("Video Feed", 1)

    # construct the class for background subtraction
    bgModel_RGB = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=16, detectShadows=False)
    frame_without_blur = None  # To store the first frame without blur

    frame_without_blur = None
    run_second_loop = True

    while True:
        ret, frame = cap.read()
        # Detect foreground
        Mask_RGB = detectForeground(frame, bgModel_RGB)
        Mask_edges = cv2.Canny(Mask_RGB, 0, 30, 3)
        # Check blurriness 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray_frame)
        
        text = "Blur"
        detect_result = model(frame)
        cv2.imshow("DNi Detector", np.squeeze(detect_result.render()))

        # Extract confidence information from the detection result
        confidence_info = (
        detect_result.pandas().xyxy[0]['confidence'] 
        if detect_result and 'confidence' in detect_result.pandas().xyxy[0]
        else 0
        )
        # if the focus measure is less than the threshold, then the image should be considered "blurry"
        if fm < fm_value:  # Replace YOUR_THRESHOLD with the desired threshold value
            cv2.putText(frame, "Blurry. Don't move", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        elif all (confidence_info < confidence_value) and (fm >= fm_value):
            # If this is the first frame without blur, store it
            cv2.putText(frame, "Not Blurry", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        elif all (confidence_info>=confidence_value) and (fm>=fm_value):
            frame_without_blur = frame.copy()
            cv2.putText(frame, "Not Blurry", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the blurriness result and DNI detection on the frame
        cv2.putText(frame, "{}: {:.2f}. Threshold: {}".format(text, fm, fm_value), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3, cv2.LINE_AA)

        #cv2.rectangle(frame, (10, 450 + 2 + 5), (10 + 2, 450 + 20 - 50),
        #            (0, 0, 0), -1)  # El 15 aquí es el espacio adicional debajo del texto

        cv2.putText(frame, "Adjust the 'fm_value' value for adjusting the threshold.".format(text, fm), (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.71, (0, 255, 255), 2, cv2.LINE_AA)


        # Display the original video frame
        cv2.imshow('Video Feed', frame)

        # # Break the loop if a frame without blur is found
        if frame_without_blur is not None:
            img = frame
            detect_result = model(frame)
            row_info = detect_result.pandas().xyxy[0].loc[0]  # Extracting information for the row with index 0
            xmin = row_info['xmin']
            ymin = row_info['ymin']
            xmax = row_info['xmax']
            ymax = row_info['ymax']
            class_id = row_info['class']
            class_name = row_info['name']
            # Now you can use these variables as needed
            print(f"xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}, class_id: {class_id}, class_name: {class_name}")
            crop = crop_image(frame, xmin, ymin, xmax, ymax)
            # Display the cropped image
            cv2.imshow('Cropped Image', crop)
            cv2.waitKey(0)
            #hola=dnifunctions.rotate(crop)
            #cv2.imshow("hola",hola)
            #cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            cv2.imwrite(path+'/images/cropped_image.jpg', crop)
            #print("save image")

            #path=os.path.abspath(os.path.join('caritas')) # Old code, replaced b SIFT
            # result=detector_caras.detectar_caras_y_comparar(crop,path)
            # print(f"\033[1;32mFaces on the image: \033[0m{result}")
                
            selected_index=None
            dni_result=match_sift_points(img)
            if dni_result==1:
                #DNI 1
                selected_index=1
            if dni_result==2:
                #DNI 2
                selected_index=2
            if dni_result==3:
                #DNI 3
                selected_index=3
            if dni_result==4:
                #DNI BACK
                MRZ_text = calculeMRZ(crop)
                if MRZ_text:
                    print(f"MRZ text:\n {MRZ_text}")
                elif not MRZ_text:
                    print("\033[1;32m" + "Empty frame or invalid dimensions." + "\033[0m")

            # Imprimir el índice seleccionado
            if selected_index==1:
                print("\033[1;32m" + "Blue DNI" + "\033[0m")
                height, width, channels = crop.shape

                #DNiNumber
                try:
                    yolo_coordinates =(0.193732, 0.927061, 0.280342, 0.145878)
                    #0.615954 0.210874 0.393162 0.121100
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    DNiNumberCrop = crop[y1:y2, x1:x2]
                    DNiNumber_Text=DatosFuncion.ocr(DNiNumberCrop)
                    if DNiNumber_Text is not None:
                        print(f"DNI: {DNiNumber_Text}")
                    else:
                        print("DNI not found correctly, try again.")
                except Exception as e:
                    print(f"DNI error: {e}")

                #Second Surname
                try:
                    yolo_coordinates =(0.453561, 0.191756, 0.338462, 0.200717)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    Surname1 = crop[y1:y2, x1:x2]
                    cv2.imshow("Surnames",Surname1)
                    cv2.waitKey(0)
                    Surname1_text=DatosFuncion.ocr(Surname1)
                    print(Surname1_text)
                except Exception as e:
                    print(f"Surnames Error: {e}")

                #Third Name
                try:
                    yolo_coordinates =(0.450142, 0.323656, 0.338462, 0.086022)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    Name = crop[y1:y2, x1:x2]
                    cv2.imshow("Name",Name)
                    cv2.waitKey(0)
                    Name_text=DatosFuncion.ocr(Name)
                    print(Name_text)
                except Exception as e:
                    print(f"Third Name Error: {e}")
                
                #Signature
                try:
                    yolo_coordinates =(0.452905, 0.822185, 0.357798, 0.319358)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    Signature = crop[y1:y2, x1:x2]
                    cv2.imshow("Signature",Signature)
                    cv2.waitKey(0)
                except Exception as e:
                    print(f"Error en Signature: {e}")
            

            elif selected_index==2:
                print("\033[1;32m" + "DNI 3.0" + "\033[0m")
                height, width, channels = crop.shape

                #DNiNumber
                try:
                    yolo_coordinates = (0.114245, 0.933201, 0.223932, 0.133598)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    DNiNumber = crop[y1:y2, x1:x2]
                    cv2.imshow("DNi",DNiNumber)
                    cv2.waitKey(0)
                    DNiNumber_text=DatosFuncion.ocr(DNiNumber)
                    print(DNiNumber_text)
                except Exception as e:
                    print(f"Error en DNiNumber: {e}")
             
                try:
                    yolo_coordinates =(0.380057, 0.179573, 0.263248, 0.203402)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    Surnames = crop[y1:y2, x1:x2]
                    cv2.imshow("Surnames",Surnames)
                    cv2.waitKey(0)
                    Surnames_text=DatosFuncion.ocr(Surnames)
                    print(Surnames_text)
                except Exception as e:
                    print(f"Error en Second Surnames: {e}")

                #Third Name
                try:
                    yolo_coordinates =(0.509971, 0.465686, 0.266667, 0.111520)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    Name = crop[y1:y2, x1:x2]
                    cv2.imshow("Name",Name)
                    cv2.waitKey(0)
                    Name_text=DatosFuncion.ocr(Name)
                    print(Name_text)
                except Exception as e:
                    print(f"Error en Third Name: {e}")

                #Signature
                try:
                    yolo_coordinates =(0.571309, 0.882894, 0.395329, 0.2280991)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    Signature = crop[y1:y2, x1:x2]
                    cv2.imshow("Signature",Signature)
                    cv2.waitKey(0)
                except Exception as e:
                    print(f"Error en Signature: {e}")

            elif selected_index==3:
                print("\033[1;32m" + "Last DNI." + "\033[0m")
                
                height, width, channels = crop.shape
                yolo_coordinates =(0.572365, 0.462194, 0.487179, 0.099915)
                y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                DNiNameCrop = crop[y1:y2, x1:x2]
                cv2.imshow("DNI NAME",DNiNameCrop)
                cv2.waitKey(0)
                try:
                    DNiName_text=DatosFuncion.ocr(DNiNameCrop)
                    if DNiName_text is not None:
                        print(f"DNI name is: {DNiName_text}")
                    else:
                        print("DNI name not found correctly, try again.")
                except Exception as e:
                    print(f"DNI name error: {DNiName_text}")

                #Surnames
                yolo_coordinates =(0.568091, 0.329875, 0.464957, 0.137720)
                y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                Surnames = crop[y1:y2, x1:x2]
                cv2.imshow("DNI SURNAMES",Surnames)
                cv2.waitKey(0)
                try:
                    DNiSurname_text=DatosFuncion.ocr(Surnames)
                    if DNiSurname_text is not None:
                        print(f"ID surname is: {DNiSurname_text}")
                    else:
                        print("DNI surname not found correctly, try again.")
                except Exception as e:
                    print(f"DNI error: {DNiSurname_text}")

                #Third dni
                try:
                    yolo_coordinates =(0.551852, 0.228610, 0.401709, 0.140421)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    IDNumber = crop[y1:y2, x1:x2]
                    cv2.imshow("DNI NUMBER ",IDNumber)
                    cv2.waitKey(0)
                    IDNumber_text=DatosFuncion.ocr(IDNumber)
                    print(IDNumber_text)
                except Exception as e:
                    print(f"Error en Third Name: {e}")

                #Signature
                try:
                    yolo_coordinates =(0.577982, 0.828024, 0.392661, 0.162321)
                    y1,y2,x1,x2 = DatosFuncion.yolo_to_absolute(height,width, yolo_coordinates)
                    Signature = crop[y1:y2, x1:x2]
                    cv2.imshow("Signature",Signature)
                    cv2.waitKey(0)
                except Exception as e:
                    print(f"Error en Signature: {e}")
            break


        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\033[1;32m" + "\nBye." + "\033[0m")
            sys.exit(1)
            break


    cv2.destroyAllWindows()
    #cap.release()

    frame_without_blur2 = None

    image_paths_elegir = [os.path.abspath(os.path.join('images','selection', 'yes.jpg')),
                            os.path.abspath(os.path.join('images','selection','no.jpg'))]

    selected_index_comparar = DatosFuncion.choose_and_return_index(image_paths_elegir, "Do you want to compare this DNi with you? Using the webcam")
    # Imprimir el índice seleccionado
    if selected_index_comparar==1:
        run_second_loop = True
        imagen_dni = crop
        print("\033[1;32m" + "Comparing DNi and person..." + "\033[0m")

        # cv2.imshow("Para comparar", imagen_dni)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cap.release()
        cap = cv2.VideoCapture(0)
        #cap2 = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #cv2.namedWindow("Video Feed", 1)
        fps = cap.get(cv2.CAP_PROP_FPS) 
        bgModel_RGB2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=16, detectShadows=False)
        frame_without_blur2 = None  # To store the first frame without blur

        while run_second_loop:
            ret2, frame2 = cap.read()

            if not ret2 or frame2 is None:
                # Puedes agregar un mensaje de error o simplemente continuar con el siguiente fotograma.
                print("Error reading frame or frame is None.")
                run_second_loop = False
                continue
            #else:
            #    print(f"Frame readed")

            #cv2.imshow("Para comparar", frame2)
            #cv2.waitKey(1)
            #print(frame2)

            # Detect foreground
            #Mask_RGB = detectForeground(frame2, bgModel_RGB2)
            #Mask_edges = cv2.Canny(Mask_RGB, 0, 30, 3)
            # Check blurriness 
            gray_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray_frame)
            
            text = "Blur"
            detect_result = model(frame2)
            cv2.imshow("Face Detector", np.squeeze(detect_result.render()))

            # Extract confidence information from the detection result
            confidence_info = (
            detect_result.pandas().xyxy[0]['confidence'] 
            if detect_result and 'confidence' in detect_result.pandas().xyxy[0]
            else 0
            )
            # if the focus measure is less than the threshold, then the image should be considered "blurry"
            if fm < fm_face_value:  # Replace YOUR_THRESHOLD with the desired threshold value
                cv2.putText(frame2, "Blurry. Don't move", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            elif fm>=fm_face_value:
                frame_without_blur2 = frame.copy()
                cv2.putText(frame2, "Not Blurry", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the blurriness result and DNI detection on the frame
            cv2.putText(frame2, "{}: {:.2f}. Threshold: {}".format(text, fm, fm_face_value), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame2, "Adjust the 'fm_face_value' value for changing the threshold.".format(text, fm), (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            # Display the original video frame
            cv2.imshow('Video Feed', frame2)

            # # Break the loop if a frame without blur is found
            if frame_without_blur2 is not None:
                img = frame2
                result=validacion.validacion(img, imagen_dni, path)
                print(f"Result: {result}")
                #cv2.destroyAllWindows()
                #cap.release()
                
                # Son la misma persona
                if result == 0:
                    print("\033[1;32m" + "They are the same person." + "\033[0m")
                    run_second_loop = False
                    break
                # NO son la misma persona
                if result == 1:
                    print("\033[1;31m" + "They are NOT the same person." + "\033[0m")
                    run_second_loop = False
                    break
                
                if result == 3:
                    print("\033[1;31m" + "Error: Invalid number of faces detected. Please make sure there is only one face in each image or adjust the 'fm_face_value' value for changing the threshold. If the photo is blurred, you should increase the value." + "\033[0m")
                    run_second_loop = False
                    break

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                run_second_loop = False




    if selected_index_comparar==2:
        print("\033[1;32m" + "Not comparing." + "\033[0m")

    # Display the first frame without blur
    cv2.imshow("Frame Without Blur", img)
   
    frame_save_path = 'images/ruta_del_frame_guardado.jpg'
    # Guarda el frame como una imagen
    cv2.imwrite(frame_save_path, frame)
    cv2.waitKey(0)

    if crop is not None:
        cv2.imshow('Cropped Image', crop)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

    print("\033[1;32m" + "\nBye." + "\033[0m")

if __name__ == '__main__':
    main()