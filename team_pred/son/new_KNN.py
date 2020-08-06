import cv2
import face_recognition
import pickle
import time, dlib

image_file = 'son_file/image/song_1.jpg'
encoding_file = 'encodings_song.pickle'
unknown_name = 'Unknown'
model_method = 'cnn'
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model = model_method)
    encodings = encode_faces(rgb, boxes)
    names = []

    #loop over the facial embeddings
    for encoding in encodings:
        #compare 함수를 통해 인코딩된 값과 현재 값 비교
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = unknown_name
        
        if True in matches:
            #매치된 것들의 인덱스 배열 후 카운트를 준다.
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            #인덱스값을 조회해 어떤이름으로 되어있는지 확인
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
           
            #가장많은 카운트의 이름을 추가
            name = max(counts, key=counts.get)
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        # name은 name 값 box = 4개의 값
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if(name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name =''

        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)

    end_time = time.time()
    process_time = end_time - start_time
    print("===A frame took {:.3f} second".format(process_time))
    cv2.imshow("Recognition", image)
    
data = pickle.loads(open(encoding_file, "rb").read())

#load the input image
image = cv2.imread(image_file)
detectAndDisplay(image)

cv2.waitKey(0)
cv2.destroyAllWindows()