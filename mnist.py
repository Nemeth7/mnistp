from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np
from PIL import Image
import time
import os

#Dataset betöltése (x= input images, y=metadata)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#2D arrayből átkonvertáljuk (reshape) 1Dbe a képeket (28*28=784pixelbe átkonvertáljuk)
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

#metadata "one-hot encoded " vektorba konvertálva, 0-9 között 10 szám ezért num_labels = 10
num_labels = 10
y_train = to_categorical(y_train, num_labels)
y_test = to_categorical(y_test, num_labels)

#512 taggal rendelkező Dense layerek (3db)
#az első layer az input (784 mert 28*28), aktiválási függvény: relu, 0.4 droput rate a túlméretezés miatt
#középső hidden layer
#3. layer az output layer, 10 kimenettel (0-9)
model = Sequential()
model.add(Dense(512, input_shape=(28 * 28,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

#model tömörítés a veszteségi funkcióval és a többivel
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

#model betanítása
batch_size = 128
epochs = 20
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

#model tesztelése
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.2f%%" % (100.0 * acc))

time.sleep(3)
os.system('cls' if os.name == 'nt' else 'clear')

while True:
    #betöltés
    image_path = input("Fájl helye: ")
    image = Image.open(image_path).resize((28, 28)).convert("L") #grayscale
    image_array = np.array(image) / 255.0 #numpy array
    image_input = image_array.reshape(1, 28 * 28) #2dből 1dbe

    #predict
    predictions = model.predict(image_input)
    predicted_label = np.argmax(predictions)
    confidence_score = np.max(predictions)

    print("Predicted label:", predicted_label)
    print("Confidence score: {:.2%}".format(confidence_score))
