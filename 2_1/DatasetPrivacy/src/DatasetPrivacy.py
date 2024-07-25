import cv2
import numpy as np
import matplotlib.pyplot as plt

data = '../Data/road.jpg'

def apply_mosaic(image, factor=0.1):
    (h, w) = image.shape[:2]
    x = int(w * factor)
    y = int(h * factor)
    return cv2.resize(cv2.resize(image, (x, y)), (w, h), interpolation=cv2.INTER_NEAREST)

def detect_face(image, detector, win_title):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        image[y:y+h, x:x+w] = apply_mosaic(face)
    
    return image

def detect_and_remove_people(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

    locations, weights = hog.detectMultiScale(image)

    remove_people_image = image.copy()
    for loc in locations:
        x, y, w, h = loc
        remove_people_image[y:y+h, x:x+w] = 0
    return remove_people_image

image = cv2.imread(data)
# 사람 제거
remove_people_image = detect_and_remove_people(image)

# 얼굴 모자이크
lbp_face_cascade = cv2.CascadeClassifier()
lbp_face_cascade.load('../data/lbpcascade_frontalface.xml')

mosaic_face_image = detect_face(image, lbp_face_cascade, 'LBP cascade face detector')

plt.figure(figsize=(12,6))
# plt.subplot(121)
# plt.title('original')
# plt.axis('off')
# plt.imshow(image[:,:,[2,1,0]])
plt.subplot(121)
plt.title('remove_people')
plt.axis('off')
plt.imshow(remove_people_image[:,:,[2,1,0]])
plt.subplot(122)
plt.title('mosaic_face')
plt.axis('off')
plt.imshow(mosaic_face_image[:,:,[2,1,0]])
plt.tight_layout()
plt.show()