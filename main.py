import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import tensorflow_hub as hub
import zipfile

#--------------------------------------------------------
# Please use Google colab to run the following code

#! wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip

zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip", "r")
zip_ref.extractall()
zip_ref.close()

train_dir = './10_food_classes_all_data/train'
test_dir = './10_food_classes_all_data/test'

data_generator = ImageDataGenerator(rescale=1 / 255.,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    rotation_range=0.2,
                                    horizontal_flip=True)

train_dt = data_generator.flow_from_directory(train_dir,
                                              batch_size=30,
                                              target_size=(224, 224),
                                              class_mode='categorical')

test_dt = data_generator.flow_from_directory(test_dir,
                                             batch_size=30,
                                             target_size=(224,224),
                                             class_mode='categorical')


#--------------------------------------------
# Build a feature extraction model using feature-extraction (transfer learning)

resnet_url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'

resnet_model = hub.KerasLayer(resnet_url, trainable=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(resnet_model)
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.summary()

performance = model.fit(train_dt,
                        epochs=6,
                        steps_per_epoch=len(train_dt),
                        validation_data=test_dt,
                        validation_steps=len(test_dir),
                        callbacks = [tf.keras.callbacks.TensorBoard( log_dir='logs',  update_freq='epoch')])

#------------------------------------------
# Print model structure
plot_model(model=model, show_shapes=True)

#------------------------------------------
# Visualize result
dt = pd.DataFrame( performance.history)
dt.plot.line()

# To create a tensorboard experiment
#!tensorboard dev upload --logdir '.' --name "ResNet50V2" --one_shot

# To delete the experiment
# !tensorboard dev delete --experiment_id 'YOUR_EXPERIMENT_ID'

# Check to see if experiments still exist
#!tensorboard dev list