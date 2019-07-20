from keras.datasets import mnist

#loading mnist dataset
(x_train,y_train), (x_test,y_test)=mnist.load_data()

#Pre-processing the data
#Since the labels ie y_test are numbers from 1-10 we need to convert them to 0's and 1's ("One hot encoded") so our cnn can understand

from keras.utils.np_utils import to_categorical
y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

#Processing the data by converting features to 0's and 1's. Therefore we need to divide by the max pixel value (255)

x_train = x_train/255
x_test = x_test/255

#Reshaping the data:The shape of our data is (60000,28,28)...we need to add colour channel.
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)

#Training the model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.summary()

#Training the model
model.fit(x_train,y_cat_train,epochs=2)


#Evaluate the model
model.metrics_names

model.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report
predictions = model.predict_classes(x_test)

print(predictions[0])
print(y_test[0])

model.save('C:/Users/Kulkarni/Downloads/Computer-Vision-with-Python/DATA/mnist_model.h5')













