import cv2, dlib
import numpy as np
from imutils import face_utils, paths
from keras.models import load_model
from tqdm import tqdm
from mtcnn import MTCNN

IMG_SIZE = (34, 26)

detector = MTCNN()
predictor = dlib.shape_predictor('models/model.dat')

model = load_model('models/2022_05_11_17_17_31.h5')


def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


# main
images = paths.list_images("dataset")

path_image = 'dataset/'

blink = 0
total = len(list(paths.list_images(path_image)))

for image in images:
    img_ori = cv2.imread(image)

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img_rgb)

    if len(faces) < 1:
        cv2.imwrite(f"results/no_face/{image.split('/')[-1]}", img)
        continue

    for face in faces:
        x, y, w, h = face['box']
        shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
        shapes = face_utils.shape_to_np(shape)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)

        # visualize
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
        cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

        cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    print("\n\n")
    if pred_l < 0.1 or pred_r < 0.1:
        cv2.imwrite(f"results/blinked/{image.split('/')[-1]}", img)
        print("blinked! ---> ", pred_l, pred_r)
        print(f"results/blinked/{image.split('/')[-1]}")
        blink += 1

    else:
        cv2.imwrite(f"results/not_blink/{image.split('/')[-1]}", img)
        print("not blinked! ---> ", pred_l, pred_r)
        print(f"results/not_blink/{image.split('/')[-1]}")

print("\n\n")
print("total: ", total)
print("blink: ", blink)
print(f"{blink}/{total}")
