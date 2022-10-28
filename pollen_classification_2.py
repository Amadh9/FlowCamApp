"""
Analyze the images extracted from the FC
Detect circles for pollens - Non Germinated
Look for germination/presence of a tail - Germinated
All others are classified as trash

Algorithm:
1- use knn for non-germ
2- if not non-germ do circle detn
3- if yes, germ else trash

"""


import cv2
import numpy as np
import os
import pickle
from skimage.feature import hog
from skimage.transform import resize
from os import path


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# measure the length of the pollen tail
#
# print("Working dir:", os.getcwd())
# print("Files in here:", os.listdir("."))
# print("Icon exists?", "hoops_icon.ico" in os.listdir("."))

transparency_thresh = 20 #20
pollen_intensity_thresh = 30.0 #30
debug = 0
knn_model = path.abspath(path.join(path.dirname(__file__), 'knn_hog_GT_85.sav'))
svm_model = path.abspath(path.join(path.dirname(__file__), 'svm_hog_trio_3_82.sav'))
knn_model2 = path.abspath(path.join(path.dirname(__file__), 'knn_hog_trio_2_77.sav'))

# knn_model ='/Users/amadh/PycharmProjects/FlowCamAssistant/knn_hog_GT_85.sav' # for non-germ
# svm_model = '/Users/amadh/PycharmProjects/FlowCamAssistant/svm_hog_trio_3_82.sav'
# knn_model2 = '/Users/amadh/PycharmProjects/FlowCamAssistant/knn_hog_trio_2_77.sav'



def classify(folder):
    dpath = os.path.join(folder, 'cropped')
    onlyfiles = [f for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f)) and not f.startswith('.')]
    garbage_path = os.path.join(folder, 'garbage')
    germ_path = os.path.join(folder, 'germinated')
    nongerm_path = os.path.join(folder, 'non_germinated')
    others_path = os.path.join(folder, 'others')

    if not os.path.exists(garbage_path):
        os.mkdir(garbage_path)

    if not os.path.exists(germ_path):
        os.mkdir(germ_path)

    if not os.path.exists(nongerm_path):
        os.mkdir(nongerm_path)

    if not os.path.exists(others_path):
        os.mkdir(others_path)

    num_garbage = 0
    num_germ = 0
    num_nongerm = 0
    num_objects = len(onlyfiles)

    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 800

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Set Convexity filtering parameters More is the convexity, the closer it is to a close circle
    params.filterByConvexity = True
    params.minConvexity = 0.2

    # Set inertia filtering parameters Objects similar to a circle has larger inertial
    # .E.g. for a circle, this value is 1, for an ellipse it is between 0 and 1, and for a line it is 0
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    kmodel = pickle.load(open(knn_model, 'rb')) # GT
    smodel = pickle.load(open(svm_model, 'rb')) # 3 class
    kmodel2 = pickle.load(open(knn_model2, 'rb')) #3 class

    for n in range(0, len(onlyfiles)):
        fn = os.path.join(dpath, onlyfiles[n])
        img = cv2.imread(fn)
        orig = img  # import image

        # first test non-germ
        resized_img = resize(img, (128, 64))

        # generating HOG features
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                            multichannel=True)


        spred = smodel.predict(fd.reshape(1, -1))[0]
        kpred2 = kmodel2.predict(fd.reshape(1, -1))[0]
        # knn for GT
        kpred = kmodel.predict(fd.reshape(1, -1))[0]
        # svm for NG

        # Detect blobs
        kernel = np.array([[0, -1, 0],
                           [-1, 7, -1],
                           [0, -1, 0]])
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        keypoints = detector.detect(image_sharp)
        number_of_blobs = len(keypoints)

        # detect multiples
        if number_of_blobs > 1: #Trash
            cv2.imwrite(garbage_path + '/' + onlyfiles[n], orig)
            num_garbage+=1
            continue

        if spred==0 and kpred2==0 and number_of_blobs==1: #NG
            cv2.imwrite(nongerm_path + '/' + onlyfiles[n], orig)
            num_nongerm+=1
            continue

        else:

            if kpred2==1 and number_of_blobs==1 and spred==1 and kpred==1:
                cv2.imwrite(germ_path + '/' + onlyfiles[n], orig)
                num_germ+=1
                continue
            elif kpred2==2 and not relaxed_blob(fn):
                cv2.imwrite(garbage_path + '/' + onlyfiles[n], orig)
                num_garbage+=1
                continue
            elif number_of_blobs==0 and kpred==0:
                cv2.imwrite(garbage_path + '/' + onlyfiles[n], orig)
                num_garbage+=1
                continue
            elif number_of_blobs==1 and kpred2==1:
                cv2.imwrite(germ_path + '/' + onlyfiles[n], orig)
                num_germ+=1
                continue
            elif number_of_blobs==1 and kpred2==0:
                cv2.imwrite(nongerm_path + '/' + onlyfiles[n], orig)
                num_nongerm+=1
                continue
            elif kpred2==0 and relaxed_blob(fn)==1:
                cv2.imwrite(nongerm_path + '/' + onlyfiles[n], orig)
                num_nongerm+=1
                continue
            #else:
            elif [spred, kpred, kpred2].count(1)>1 and number_of_blobs==1:
                cv2.imwrite(germ_path + '/' + onlyfiles[n], orig)
                num_germ+=1
                continue
            elif [spred, kpred, kpred2].count(0)>1 and relaxed_blob(fn)==1:
                cv2.imwrite(nongerm_path + '/' + onlyfiles[n], orig)
                num_nongerm+=1
                continue
            elif [spred, kpred, kpred2].count(1)>=2 and relaxed_blob(fn)==1:
                # implement a more relaxed blob detection
                cv2.imwrite(germ_path + '/' + onlyfiles[n], orig)
                num_germ+=1
                continue
            elif spred==1 and number_of_blobs==0 and relaxed_blob(fn)==1:
                cv2.imwrite(germ_path + '/' + onlyfiles[n], orig)
                num_germ+=1
                continue
            elif kpred2==1:
                if (relaxed_blob(fn)):
                    cv2.imwrite(others_path + '/' + onlyfiles[n], orig)
                    continue
                else:
                    cv2.imwrite(garbage_path + '/' + onlyfiles[n], orig)
                    num_garbage+=1
                    continue

            else:
                if kpred2==0 and relaxed_blob(fn):
                    cv2.imwrite(nongerm_path + '/' + onlyfiles[n], orig)
                    num_nongerm+=1

                else:
                    cv2.imwrite(garbage_path + '/' + onlyfiles[n], orig)
                    num_garbage+=1

    lines = "Num Objects Detected   : " + str(num_objects), \
            "Num Germinated         : " + str(num_germ), \
            "Num Non-Germinated     : " + str(num_nongerm), \
            "Num Garbage            : " + str(num_garbage)

    fn = folder + '/stats.txt'
    with open(fn, 'w') as f:
        for l in lines:
            f.writelines(l)
            f.write('\n')
    f.close()
    return




def relaxed_blob(test_img):

    img = cv2.imread(test_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # change to grayscale
    img = cv2.bilateralFilter(img_gray, 8, 150, 150)  # 15,80,80 #10,50,75

    kernel = np.array([[0, -1, 0],
                       [-1, 7, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    # Set our filtering parameters
    # Initialize parameter settiing using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea =700

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Set Convexity filtering parameters More is the convexity, the closer it is to a close circle
    params.filterByConvexity = True
    params.minConvexity = 0.1 #0.1

    # Set inertia filtering parameters Objects similar to a circle has larger inertial
    # .E.g. for a circle, this value is 1, for an ellipse it is between 0 and 1, and for a line it is 0
    params.filterByInertia = True
    params.minInertiaRatio = 0.01#0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)


    # Detect blobs
    keypoints = detector.detect(image_sharp)

    # # Draw blobs on our image as red circles
    # blank = np.zeros((1, 1))
    # blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    return number_of_blobs


def crop_boxes(mypath,tosave):

    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and not f.startswith('.') \
                 and f.endswith(('.tif', '.jpg', '.jpeg', '.bmp', '.png'))]
    images = np.empty(len(onlyfiles), dtype=object)
    idx = 0
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(os.path.join(mypath, onlyfiles[n]))

        gwash = images[n]  # import image

        gwashBW = cv2.cvtColor(gwash, cv2.COLOR_RGB2GRAY)  # change to grayscale

        height = np.size(gwash, 0)
        width = np.size(gwash, 1)

        ret, thresh1 = cv2.threshold(gwashBW, 41, 255, cv2.THRESH_BINARY)

        kernel = np.ones((1, 1), np.uint8)

        erosion = cv2.erode(thresh1, kernel, iterations=31)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = []  # list to hold all areas

        if not os.path.exists(tosave):
            os.mkdir(tosave)
        for i, contour in enumerate(contours):
            ar = cv2.contourArea(contour)
            areas.append(ar)
            cnt = contour
            (x, y, w, h) = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 600 and cv2.contourArea(cnt) < (height * width):
                if hierarchy[0, i, 3] == -1:
                    cv2.rectangle(gwash, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    new_img = gwash[y:y + h, x:x + w]
                    idx += 1
                    cv2.imwrite(tosave + str(idx) + '.png', new_img)

    # print("here")
    return



