import cv2
import face_recognition
import pickle
import dlib 

dataset_paths = ['son_file/image/song-test/']
names = ['Song']
number_images = 10
image_type = '.jpg'
encoding_file = 'encodings_song.pickle'
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
model_method = 'cnn'

knownEncodings =[]
knownNames = []

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

#사진에 대한 이름 분류
for (i, dataset_path) in enumerate(dataset_paths):
    name = names[i]
    print("names:", name)
    #파일음 이름 선정
    for idx in range(number_images):
        file_name = dataset_path + str(idx+1) + image_type

    image = cv2.imread(file_name)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model = model_method)

    #인식후 해야하는 인코딩
    encodings = encode_faces(rgb, boxes)
     
    for encoding in encodings:
        print(file_name, name, encoding)
        knownEncodings.append(encoding)
        knownNames.append(name)

#인코딩 값 저장
data = {"encodings": knownEncodings, "names": knownNames}
print("names:", name)
f = open(encoding_file, "wb")
f.write(pickle.dumps(data))
f.close()
