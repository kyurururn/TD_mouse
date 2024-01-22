import imutils
import numpy as np
import cv2
import pyautogui
import keyboard

cap = cv2.VideoCapture(0)

prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

face_coordinates = []

move_enabled = True  # 追加: マウス移動を制御するフラグ

while True:
    ret, frame = cap.read()
    img = imutils.resize(frame, width=400)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    face_coordinates.clear()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face_coordinates.append((startX, startY, endX, endY))
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    print("Detected face coordinates:", (startX + endX) // 2, (startY + endY) // 2)
    cv2.imshow("Face Detection", img)

    k = cv2.waitKey(1) & 0xff
    if not keyboard.is_pressed("s"):
        pyautogui.moveTo(1600 - (startX + endX) // 2 * 4, (startY + endY) // 2 * 4, _pause=False)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
