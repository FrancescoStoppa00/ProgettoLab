from cgi import test
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

#DIVIDO IL MIO SET
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
print(f'ho {train_size} batch per il test set')
print(f'ho {val_size} batch per il validation set')
#print(f'ho {test_size} batch per il test set')

#dico quanti batch sono necessari per il mio training, e dopo quanti batch dovrò utilizzzare gli altri data
train = data.take(train_size)
val = data.skip(train_size).take(val_size) 
#test = data.skip(train_size+val_size).take(test_size)

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
model.add(Dense(10, activation='softmax'))#softmax viene utilizzato per le classificazioni multi-class
#utilizzo come funzione di loss una classificazione binaria, e voglio tener d'occhio l'accuracy della nostra rete
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(model.summary())

#TRAIN
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
history = model.fit(train, epochs=10, validation_data=val, callbacks = [tensorboard_callback], verbose=2)

#EVALUATE PERFORMANCE 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#TEST real time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset=20
imgSize = 400
folder = 'data123'
counter = 0
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    cv2.imshow("imgOutput", imgOutput)
    hands, img = detector.findHands(img)
    if len(hands)>1:
            hand1 = hands[0]
            hand2 = hands[1]
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']
            #devo fare in modo che le immagini crop siano tutte della stessa dimensione
            #e fare in modo che la mano venga inserita al centro della nostra immagine
            imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgWhite2 = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop1 = img[y1-offset:y1+h1+offset, x1-offset:x1+w1+offset]
            imgCrop2 = img[y2-offset:y2+h2+offset, x2-offset:x2+w2+offset]

            aspectRatio1 = h1/w1
            aspectRatio2 = h2/w2
            if aspectRatio1 > 1:
                k1 = imgSize/h1
                #qualsiasi valore con la virgola viene approssimato all'intero superiore
                wCal1 = math.ceil(k1*w1)
                imgResize1 = cv2.resize(imgCrop1, (wCal1, imgSize))
                #devo trovare lo spazio per poter spostare l'immagine al centro
                wGap1 = math.ceil((imgSize-wCal1)/2)
                imgWhite1[ : , wGap1:wCal1+wGap1] = imgResize1

            else:
                k1 = imgSize/h1
                hCal1 = math.ceil(k1*h1)
                imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal1))
                hGap1 = math.ceil((imgSize-hCal1)/2)
                imgWhite1[hGap1:hCal1+hGap1, : ] = imgResize1

            #cv2.imshow("ImageCrop1", imgCrop1)

            if aspectRatio2 > 1:
                k2 = imgSize/h2
                #qualsiasi valore con la virgola viene approssimato all'intero superiore
                wCal2 = math.ceil(k2*w2)
                imgResize2 = cv2.resize(imgCrop2, (wCal2, imgSize))
                #devo trovare lo spazio per poter spostare l'immagine al centro
                wGap2 = math.ceil((imgSize-wCal2)/2)
                imgWhite2[ : , wGap2:wCal2+wGap2] = imgResize2

            else:
                k2 = imgSize/h2
                hCal2 = math.ceil(k2*h2)
                imgResize2 = cv2.resize(imgCrop2, (imgSize, hCal2))
                hGap2 = math.ceil((imgSize-hCal2)/2)
                imgWhite2[hGap2:hCal2+hGap2, : ] = imgResize2

            #cv2.imshow("ImageCrop2", imgCrop2)
            
            
        #testo la mia immagine finale
            resize1 = tf.image.resize(imgWhite1, (400,400))
            resize2 = tf.image.resize(imgWhite2, (400,400))
                #inserisco la mia immagine in un array con dimensione essatta in modo da passarla al modello
            yhat1 = model.predict_classes(np.expand_dims(resize1/255, 0), batch_size=1)
            yhat2 = model.predict_classes(np.expand_dims(resize2/255, 0), batch_size=1)
            cv2.putText(imgOutput, str(yhat1[0]), (x1, y1-offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0 ,255), 2)
            cv2.putText(imgOutput, str(yhat2[0]), (x2, y2-offset), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255 ,255), 2)
    elif hands:
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

        #cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        
        #testo la mia immagine finale
        resize = tf.image.resize(imgWhite, (400,400))
        #incapsulo la mia immagine in un array
        yhat = model.predict_classes(np.expand_dims(resize/255, 0), batch_size=1)
        cv2.putText(imgOutput, str(yhat[0]), (x, y-offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0 ,255), 2)
    cv2.imshow("imgOutput", imgOutput)
            
    key = cv2.waitKey(1)
    if key == ord('e'):
        cap.release()
        cv2.destroyAllWindows()
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)