# Transfer-learning by feature-extraction

In this repository,we implement a sample code for feature-extraction in neural-network.

The feature-extraction means, we can use an "already-trained" model from Tensorflow-Hub and use it in our model.
But, we must change the output layer inorder to adapt the model to our own problem.
In the repository we use the *resnet_v2_50*.

<pre>resnet_url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'

resnet_model = hub.KerasLayer(resnet_url, trainable=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(resnet_model)
model.add(Dense(10, activation='softmax')) </pre>

--------------------

## Callbacks

We use tensorboard to present the neural-network results:

The callback setting is as following:

`callbacks = [tf.keras.callbacks.TensorBoard( log_dir='logs',  update_freq='epoch')]`

The `log_dir` is the directory to save the logs file and the `update_freq` declares "how often" the new loss and metrics should be recorded.

### To create a tensorboard experiment
`!tensorboard dev upload --logdir '.' --name "ResNet50V2" --one_shot`

### To delete the experiment
`!tensorboard dev delete --experiment_id 'YOUR_EXPERIMENT_ID'`

### Check to see if experiments still exist
`!tensorboard dev list`

---------------------------

## Print Neural-Network structure

We generate a png picture of the model's structure using the following code:

<pre>from tensorflow.keras.utils import plot_model
plot_model(model)</pre>


![Feature Extraction by Neural-Network](https://user-images.githubusercontent.com/4312244/152534622-559b70fe-fcdb-41bb-b294-7c98d2ca5881.png)
