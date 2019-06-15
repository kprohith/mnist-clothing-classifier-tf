# Clothing Classifier

## Basic classifier that is trained to classify images of clothing into [10 categories](https://github.com/zalandoresearch/fashion-mnist#labels)

### Trained using FashionMNIST data set consisting of 60,000 labelled images, each of 25px x 25px size

### Tested on 10,000 labelled images, which werent part of the training set.

## This classifier has close to __~80% accuracy__ on grayscale imaages of 25px x 25px size.

### Uses [tensorflow](https://www.tensorflow.org/), [keras api](https://keras.io/), [numpy](https://www.numpy.org/),[matplotlib](https://matplotlib.org/) and the [Fashion MNIST datsset](https://github.com/zalandoresearch/fashion-mnist).

### Deployed [tf.train.AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer), and the [sparse_categorical_crossentropy](https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/) loss function while compiling model.
'''python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 ''' 
  
### Uses three neural layers:
'''python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
'''