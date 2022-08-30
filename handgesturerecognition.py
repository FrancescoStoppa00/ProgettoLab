import tensorflow as tf
import os
import cv2

#righe necessarie per evitare errori causati da grandi dataset e dal consumo di risorse
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#creo percorso verso le mie sottocategorie
data_dir = 'data123'
print(os.listdir(data_dir))

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# #classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# offset=20
# imgSize = 400

# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)

#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         #devo fare in modo che le immagini crop siano tutte della stessa dimensione
#         #e fare in modo che la mano venga inserita al centro della nostra immagine
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
#         imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

#         aspectRatio = h/w
#         if aspectRatio > 1:
#             k = imgSize/h
#             #qualsiasi valore con la virgola viene approssimato all'intero superiore
#             wCal = math.ceil(k*w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             #devo trovare lo spazio per poter spostare l'immagine al centro
#             wGap = math.ceil((imgSize-wCal)/2)
#             imgWhite[ : , wGap:wCal+wGap] = imgResize
#             # prediction, index = classifier.getPrediction(imgWhite)
#             # print(prediction, index)
#             label = check_accuracy(transforms.ToTensor(imgWhite), model)
#             print(label)

#         else:
#             k = imgSize/h
#             hCal = math.ceil(k*h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             hGap = math.ceil((imgSize-hCal)/2)
#             imgWhite[hGap:hCal+hGap, : ] = imgResize
#             # prediction, index = classifier.getPrediction(imgWhite)
#             # print(prediction, index)
#             label = check_accuracy(transforms.ToTensor(imgWhite), model)
#             print(label)

#         #cv2.putText(imgOutput, labels[index], (x, y-offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0 ,255), 2)
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
    

#     cv2.imshow("image", imgOutput)
#     cv2.waitKey(1)