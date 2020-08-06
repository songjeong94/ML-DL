import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('son_file/model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):        # 페이스 디텍터
    dets = detector(img, 1) 

    if len(dets) == 0:  #얼굴 못찾으면 빈값을 반환
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], [] #결과물 저장 변수
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int) #결과물 저장 변수(랜드마크 68개의 점)
    for k, d in enumerate(dets): #얼굴마다 루프를 돈다.
        rect = ((d.left(), d.top()), (d.right(), d.bottom())) # 얼굴인식후  4가지 좌표 변수 지정
        rects.append(rect) 

        shape = sp(img, d) #랜드마크 구하기 이미지와 d =사각형을 넣으면 shape에 68개의 점 생성

        # convert dlib shape to numpy array
        for i in range(0, 68):  #랜드마크 결과들을 쉐이프에 쌓는다.
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return rects, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes: #랜드마크들의 배열 집합
        face_descriptor = facerec.compute_face_descriptor(img, shape) #(전체이미지, 랜드마크)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

#Compute Saved Face Descriptions

img_paths = {
    'son': 'son_file/image/son-test/1.jpg',
    'kang': 'son_file/image/kang-test/10.jpg',
    'tedy': 'son_file/image/tedy-test/1.jpg'
}

descs = {
    'son': None,
    'kang': None,
    'tedy': None
}

for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb) #얼굴찾아서 랜드마크를 받아온다.
    descs[name] = encode_faces(img_rgb, img_shapes)[0] #인코더에 각사람의 이미지와 랜드마크를 저장

np.save('descs.npy', descs)
print(descs)

#Compute Input

img_bgr = cv2.imread('son_file/image/soccer_01.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)

#Visualize Output

fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):

    found = False
    for name, saved_desc in descs.items():
        dist = np.linalg.norm([desc] - saved_desc, axis=1) #유클리디안 거리

        if dist < 0.6:
            found = True

            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                    color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

            break

    if not found:
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.savefig('output.png')
plt.show()