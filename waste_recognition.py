# importing libraries
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# preprocessing images
def preprocessing(img):      
    
    # normalizing values to 0-1 from 0-255
    img = img.astype("float32")

    img = img / 255            
   
    return img


# path to the dataset with 8 classes 
dataset = "Train_Waste" 

image_size = (200, 200, 3)

# counter
count = 0

# list to store images
images = []

# list to store class numbers
class_number = []

myList = os.listdir(dataset)
print("Total Classes Detected:", len(myList))

# size: 8 classes
classes_size = len(myList)

# parsing each sub folder in dataset
for i in range(0, classes_size):
    
    sub_folder = os.listdir(dataset + "/" + str(count))
    
    # parsing each image in each sub folder
    for image in sub_folder:
        
        # preprocessing current image
        current_image = cv2.imread(dataset + "/" + str(count) + "/" + image)
        current_image = cv2.resize(current_image, (200, 200))  

        # saving each preprocessed image
        images.append(current_image)

        # stroring class numbers: 0-7
        class_number.append(count)
    
    print(count, end=" ")
    
    count += 1

print(" ")

# conversting all images to np-array
images = np.array(images)

# converting class numbers: 0-7 to np-array
class_number = np.array(class_number)
 
# splitting data to training data 80 % and testing data 20 %
X_train, X_test, y_train, y_test = train_test_split(images, class_number, test_size=0.2, random_state=42)

# parsing and preprocessing all images
X_train = np.array(list(map(preprocessing, X_train)))

X_test = np.array(list(map(preprocessing, X_test)))

# adding a depth, to 3
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], X_train.shape[2], 3)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)

# generating more data for the model
data_generator = ImageDataGenerator (
    
        width_shift_range=0.1,   
        height_shift_range=0.1,
        zoom_range=0.2,  
        shear_range=0.1,  
        rotation_range=10
) 

# fitting process
data_generator.fit(X_train)

# one-hot encoding
y_train = to_categorical(y_train,classes_size)

y_test = to_categorical(y_test, classes_size)
 
# creating CNN model
model = Sequential()

# 1st CNN layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(200, 200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd CNN layer
model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd CNN layer
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th CNN layer
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

# fully connected 1st layer
model.add(Dense(units=256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# fully connected 2nd layer
model.add(Dense(units=512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(classes_size))
model.add(Activation("softmax"))

# using callbacks 
checkpoint = ModelCheckpoint("waste_management_model.h5", monitor='accuracy', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(

    monitor='accuracy',
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights=True
)

reduce_learning_rate = ReduceLROnPlateau(

    monitor='accuracy',
    factor=0.2,
    patience=3,
    verbose=1,
    min_delta=0.0001
)

callbacks_list = [checkpoint, early_stopping, reduce_learning_rate]

# compiling model
model.compile(

    Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# fitting process
history = model.fit( 
    
    data_generator.flow(X_train, y_train, batch_size=64),
    epochs=12,
    shuffle=1,
    callbacks=callbacks_list
)
 
# saving model
model.save("waste_management_model.h5")