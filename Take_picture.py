import os

import cv2

print("Use the space bar to take a picture, 3 per person is enough")
print("Use the Escape key to stop")
print("Images are stored in /KnownImages")

name = input("Please give the name of the person")

if not os.path.exists('KnownImages'):
    os.makedirs('KnownImages')

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Take picture, look at the camera please", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = f"KnownImages/{name}_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
