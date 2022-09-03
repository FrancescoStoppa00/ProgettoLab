import tensorflow as tf
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
#Conv2D rappresenta la parte di convoluzione sulle immagini
#maxpooling2d serve per condensare tutti i dati e riportare quelli più importanti
import math
from cvzone.HandTrackingModule import HandDetector
import time

#righe necessarie per evitare errori causati da grandi dataset e dal consumo di risorse
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#creo percorso verso le mie sottocategorie
data_dir = 'data123'

#LOAD DATA
#crea un dataset senza utilizzare label o creare effettivamente classi
#va a ridimensionare tutte le immagini del dataset
data = tf.keras.utils.image_dataset_from_directory('data123', image_size = (400,400))

#PREPROCESSING DATA
#fase in cui modifico le immagini in range che vanno da 0 a 1 ainvece che da 0 a 255 
#ciò permette al mio modello di addestrarsi più velocemente e produrre risultati migliori
data = data.map(lambda x,y: (x/255.0,y)) #x rappresenta le immagini, y le labels
#rendo il dataset un iteratore numpy in modo da poter 'viaggiare' nel dataset
#il batch che è composto da due parti una prima rappresenta le immagine, nel nostro caso sono 32 poichè batch_size=32 per default
# #la seconda la label, quindi un array di 32 numeri dove ogni numero ci dice l'etichetta dell'immagine corrispondente
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
print(batch[0])
print(batch[0].max())
print(batch[0].min()) 

#DIVIDO IL MIO SET
train_size = int(len(data)*.7)
val_size = int(len(data)*.3)

#dico quanti batch sono necessari per il mio training, e dopo quanti batch dovrò utilizzzare gli altri data
train = data.take(train_size)
val = data.skip(train_size).take(val_size) 

#BUILD DEEP LEARNING MODEL
model = Sequential()
#aggiungo i livelli, 3 blocchi di convoluzione, uno strato flatten e due di dense
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(400,400,3))) #primo bloccco costituito da input, filtro costituito da 3x3 pixel e fa un passo di un pixel alla volta
model.add(MaxPooling2D()) #riduce i dati dell'immagine
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten()) #appiattisco i dati, ovvero converto tutto in un array ad una dimensione 
model.add(Dense(256, activation='relu'))#256 sono i neuroni presenti in questo livello
model.add(Dense(4, activation='softmax'))#softmax viene utilizzato per le classificazioni multi-class
#utilizzo come funzione di loss una classificazione binaria, e voglio tener d'occhio l'accuracy della nostra rete
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(model.summary())

#TRAIN
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
history = model.fit(train, epochs=10, validation_data=val, callbacks = [tensorboard_callback])

#EVALUATE PERFORMANCE 
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

#TEST real time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset=20
imgSize = 400
folder = 'data123/3'
counter = 0
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        #devo fare in modo che le immagini crop siano tutte della stessa dimensione
        #e fare in modo che la mano venga inserita al centro della nostra immagine
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            #qualsiasi valore con la virgola viene approssimato all'intero superiore
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            #devo trovare lo spazio per poter spostare l'immagine al centro
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[ : , wGap:wCal+wGap] = imgResize

        else:
            k = imgSize/h
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, : ] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    #testo la mia immagine finale
        resize = tf.image.resize(imgWhite, (400,400))
        yhat = model.predict_classes(np.expand_dims(resize/255, 0))
        print(f'il valore predetto è {yhat}')
    cv2.imshow("image", imgOutput)
    #cv2.putText(imgOutput, str(yhat[0]), (x, y-offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0 ,255), 2)
    key = cv2.waitKey(1)
    # if key == ord("s"):
    #     counter += 1
    #     cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
    #     print(counter)