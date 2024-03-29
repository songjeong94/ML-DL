import cv2
import face_recognition
import pickle

dataset_paths = ['son_file/image/son-align/', 'son_file/image/tedy-align/']
names = ['Son', 'Tedy']
number_images = 10
image_type = '.jpg'
encoding_file = 'encodings2.pickle'

model_method = 'dnn'

knownEncodings =[]
knownNames = []

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
    encodings = face_recognition.face_encodings(rgb, boxes)
     
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
