import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model

#--------------------------------------------------------
# Please use Google colab to run the following code


#------------------------------
# Get data

# ! wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip

zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip", "r")
zip_ref.extractall()
zip_ref.close()

train_dir = './10_food_classes_all_data/train/'
test_dir = './10_food_classes_all_data/train/'

train_data = image_dataset_from_directory(directory=train_dir,
                                          image_size = (224,224),
                                          label_mode = "categorical",
                                          batch_size =30 )

test_data = image_dataset_from_directory(directory = test_dir,
                                         image_size = (224,224),
                                         label_mode = 'categorical',
                                         batch_size = 30)

#---------------------------------
# Build the model

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = Input(shape=(224,224,3),name='input_layer')
x = base_model(inputs)
#print(f"Shape after base_model: {x.shape}")

x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
#print(f"After GlobalAveragePooling2D(): {x.shape}")

outputs = Dense(10,activation='softmax')(x)
model = tf.keras.Model(inputs,outputs)

#-----------------------------------
# Compile and fit

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(train_data,epochs=4,validation_data = test_data)

#-------------------------------------
# Get the structure

#model.summary()
#plot_model(model,show_shapes=True)
