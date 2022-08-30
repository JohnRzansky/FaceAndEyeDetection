import cv2
import os

images = []
for filename in os.listdir("Images"):
    img = cv2.imread(os.path.join("Images", filename))
    dimensions = img.shape
    if dimensions[0] > 487:
        height = 487
        width = int((dimensions[1]*487)/dimensions[0])
        img_resized = cv2.resize(img, (width, height))
    else:
        img_resized = img
    if img_resized is not None:
        images.append(img_resized)

face_detection = cv2.CascadeClassifier("Haarcascades\haarcascade_frontalface_default.xml")
eye_detection = cv2.CascadeClassifier("Haarcascades\haarcascade_eye.xml")

for image in images:
    faces = face_detection.detectMultiScale(image, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 3)
        cv2.imshow("Multi Face and Eyes Detection", image)
        cv2.waitKey()

        face_region = image[y:y+h, x:x+w]
        eye = eye_detection.detectMultiScale(face_region, 1.1, 5)

        for (eye_x, eye_y, eye_w, eye_h) in eye:
            cv2.rectangle(face_region, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 255), 1)
            cv2.imshow("Multi Face and Eyes Detection", image)
            cv2.waitKey()

    cv2.destroyAllWindows()



