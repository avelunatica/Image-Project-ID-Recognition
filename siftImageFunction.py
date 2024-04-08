import os
import cv2
import numpy as np

def match_sift_points(img):
    # Initialize SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.05, edgeThreshold=50)

    # Read the images to compare
    IDCardTemplate1 = cv2.imread(os.path.abspath(os.path.join(os.getcwd(),'images', 'DNItypes', 'DNI Azul.jpg')), cv2.IMREAD_GRAYSCALE)
    IDCardTemplate2 = cv2.imread(os.path.abspath(os.path.join(os.getcwd(),'images', 'DNItypes', 'Front3-0.jpg')), cv2.IMREAD_GRAYSCALE)
    IDCardTemplate3 = cv2.imread(os.path.abspath(os.path.join(os.getcwd(),'images', 'DNItypes', 'DNi Ultimo.jpg')), cv2.IMREAD_GRAYSCALE)
    IDCardTemplate4 = cv2.imread(os.path.abspath(os.path.join(os.getcwd(),'images', 'DNItypes', 'Back.jpg')), cv2.IMREAD_GRAYSCALE)
    IDCardTemplate5 = cv2.imread(os.path.abspath(os.path.join(os.getcwd(),'images', 'DNItypes', 'BackOld.jpg')), cv2.IMREAD_GRAYSCALE)  # Añade el quinto template
    IDCardTemplate6 = cv2.imread(os.path.abspath(os.path.join(os.getcwd(),'images', 'DNItypes', 'BackNew.jpg')), cv2.IMREAD_GRAYSCALE)

    # Resize all templates to have 425x270 pixels
    IDCardTemplate1 = cv2.resize(IDCardTemplate1, (159, 101))
    IDCardTemplate2 = cv2.resize(IDCardTemplate2, (159, 101))
    IDCardTemplate3 = cv2.resize(IDCardTemplate3, (159, 101))
    IDCardTemplate4 = cv2.resize(IDCardTemplate4, (159, 101))
    IDCardTemplate5 = cv2.resize(IDCardTemplate5, (159, 101))
    IDCardTemplate6 = cv2.resize(IDCardTemplate6, (159, 101))


    # Stack all templates for visualization (inefficient)
    IDCardTemplate = np.vstack((IDCardTemplate1, IDCardTemplate2, IDCardTemplate3, IDCardTemplate4, IDCardTemplate5, IDCardTemplate6))

    # Detect SIFT features in the ID card templates
    IDCard_keypoints1, IDCard_descriptors1 = sift.detectAndCompute(IDCardTemplate1, None)
    IDCard_keypoints2, IDCard_descriptors2 = sift.detectAndCompute(IDCardTemplate2, None)
    IDCard_keypoints3, IDCard_descriptors3 = sift.detectAndCompute(IDCardTemplate3, None)
    IDCard_keypoints4, IDCard_descriptors4 = sift.detectAndCompute(IDCardTemplate4, None)
    IDCard_keypoints5, IDCard_descriptors5 = sift.detectAndCompute(IDCardTemplate5, None)  # Añade el quinto template
    IDCard_keypoints6, IDCard_descriptors6 = sift.detectAndCompute(IDCardTemplate6, None)

    # Detect SIFT features in the combined ID card templates for visualization (inefficient)
    IDCard_keypoints, IDCard_descriptors = sift.detectAndCompute(IDCardTemplate, None)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Convert the frame to grayscale. SIFT only works on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if max(gray.shape[:2]) > 500:
            gray = cv2.resize(gray, (0, 0), fx=500/max(gray.shape[:2]), fy=500/max(gray.shape[:2]))
            img = cv2.resize(img, (0, 0), fx=500/max(img.shape[:2]), fy=500/max(img.shape[:2]))	
    
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Match the features using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches1 = flann.knnMatch(IDCard_descriptors1, descriptors, k=2)
    matches2 = flann.knnMatch(IDCard_descriptors2, descriptors, k=2)
    matches3 = flann.knnMatch(IDCard_descriptors3, descriptors, k=2)
    matches4 = flann.knnMatch(IDCard_descriptors4, descriptors, k=2)
    matches5 = flann.knnMatch(IDCard_descriptors5, descriptors, k=2)  # Añade el quinto template
    matches6 = flann.knnMatch(IDCard_descriptors6, descriptors, k=2)
    matches = flann.knnMatch(IDCard_descriptors, descriptors, k=2)  # (inefficient) only for visualization

    # Filter matches using the Lowe's ratio test
    good_matches1 = [m for m, n in matches1 if m.distance < 0.5 * n.distance]
    good_matches2 = [m for m, n in matches2 if m.distance < 0.5 * n.distance]
    good_matches3 = [m for m, n in matches3 if m.distance < 0.5 * n.distance]
    good_matches4 = [m for m, n in matches4 if m.distance < 0.5 * n.distance]
    good_matches5 = [m for m, n in matches5 if m.distance < 0.5 * n.distance]  # Añade el quinto template
    good_matches6 = [m for m, n in matches6 if m.distance < 0.5 * n.distance]

    # Compare the number of good matches
    template_number = 0
    max_matches = max(len(good_matches1), len(good_matches2), len(good_matches3), len(good_matches4), len(good_matches5), len(good_matches6))  # Añade el quinto template
    #print(f"Max matches: {max_matches}")
   


    # Overlay the template number on the image if it has enough matches
    if max_matches > 5:
        cv2.putText(img, str(template_number), (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Need to draw only good matches, so create a mask (inefficient)
    good_matches = [[0, 0] for i in range(len(matches))]

    # Good matches (inefficient)
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches[i] = [1, 0]

    # Draw matches (inefficient)
    match_frame = cv2.drawMatchesKnn(IDCardTemplate,
                                        IDCard_keypoints,
                                        img,
                                        keypoints,
                                        matches,
                                        outImg=None,
                                        # matchColor=(0, 155, 0),
                                        # singlePointColor=(0, 255, 255),
                                        matchesMask=good_matches,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                        )

    # Display the image with matches
    cv2.imshow("Matches", match_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('C:/Users/Usuario/Desktop/P3 AIMV/images/cropped_image.jpg')
    match_sift_points(img)