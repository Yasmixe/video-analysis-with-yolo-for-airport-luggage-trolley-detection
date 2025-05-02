import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
"""
Mask R-CNN
Multi-GPU Support for Keras.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Ideas and a small code snippets from these sources:
https://github.com/fchollet/keras/issues/2436
https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Use TF 1.x style APIs
class ParallelModel(KM.Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """
        
    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        super(ParallelModel, self).__init__() 
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
    
    # Réinitialisation correcte des entrées/sorties
        self._set_inputs(self.inner_model.inputs)
        self.outputs = merged_outputs
       
        

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                           self.inner_model.inputs)}

        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    zipped_inputs = zip(self.inner_model.input_names,
                                        self.inner_model.inputs)
                    inputs = [
                        KL.Lambda(lambda s: input_slices[name][i],
                                  output_shape=lambda s: (None,) + s[1:])(tensor)
                        for name, tensor in zipped_inputs]
                    # Create the model replica and get the outputs
                    outputs = self.inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                # Concatenate or average outputs?
                # Outputs usually have a batch dimension and we concatenate
                # across it. If they don't, then the output is likely a loss
                # or a metric value that gets averaged across the batch.
                # Keras expects losses and metrics to be scalars.
                if K.int_shape(outputs[0]) == ():
                    # Average
                    m = KL.Lambda(lambda o: tf.add_n(o) / len(outputs), name=name)(outputs)
                else:
                    # Concatenate
                    m = KL.Concatenate(axis=0, name=name)(outputs)
                merged.append(m)
        return merged


if __name__ == "__main__":
    # Testing code below. It creates a simple model to train on MNIST and
    # tries to run it on 2 GPUs. It saves the graph so it can be viewed
    # in TensorBoard. Run it as:
    #
    # python3 parallel_model.py

    import os
    import numpy as np
    import tensorflow.keras.optimizers
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow import keras

    GPU_COUNT = 2

    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    def build_model(x_train, num_classes):
        # Reset default graph. Keras leaves old ops in the graph,
        # which are ignored for execution but clutter graph
        # visualization in TensorBoard.
        

        inputs = KL.Input(shape=x_train.shape[1:], name="input_image")
        x = KL.Conv2D(32, (3, 3), activation='relu', padding="same",
                      name="conv1")(inputs)
        x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
                      name="conv2")(x)
        x = KL.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
        x = KL.Flatten(name="flat1")(x)
        x = KL.Dense(128, activation='relu', name="dense1")(x)
        x = KL.Dense(num_classes, activation='softmax', name="dense2")(x)

        return KM.Model(inputs=inputs, outputs=x, name="digit_classifier_model")
    # Load MNIST Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # Build data generator and model
    datagen = ImageDataGenerator()
    model = build_model(x_train, 10)

    # Add multi-GPU support.
    model = ParallelModel(model, GPU_COUNT)

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=5.0)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    # Train
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=50, epochs=10, verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[keras.callbacks.TensorBoard(log_dir=MODEL_DIR,
                                               write_graph=True)]
    )

# Root directory of the project
ROOT_DIR = "C:\\Users\\yasmi\\Documents\\detection\\Mask-R-CNN-using-Tensorflow2-main\\Mask-R-CNN-using-Tensorflow2-main"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + car and truck

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "chariot")

        # Train or validation dataset?
        assert subset in ["train", "valid"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open('C:\\Users\\yasmi\\Documents\\detection\\Mask-R-CNN-using-Tensorflow2-main\\Mask-R-CNN-using-Tensorflow2-main\\train1.json'))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['label'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"chariots": 1}

            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
     
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
    
        info = self.image_info[image_id]
        if info["source"] != "object":
         return super(self.__class__, self).load_mask(image_id)

    # Get image dimensions
        height, width = info["height"], info["width"]
    
    # Filter out non-polygon shapes
        polygons = [p for p in info["polygons"] if p['name'] == 'polygon']
    
        if not polygons:
            return np.zeros([height, width, 0], dtype=np.uint8), np.array([], dtype=np.int32)

        mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)
        num_ids = []
    
        for i, p in enumerate(polygons):
         # Clamp coordinates to image boundaries
            y_coords = np.clip(p['all_points_y'], 0, height-1)
            x_coords = np.clip(p['all_points_x'], 0, width-1)
        
            rr, cc = skimage.draw.polygon(y_coords, x_coords, shape=(height, width))
            mask[rr, cc, i] = 1
            num_ids.append(info['num_ids'][i])
    
        return mask, np.array(num_ids, dtype=np.int32)
    

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom("C:\\Users\\yasmi\\Documents\\detection\\Mask-R-CNN-using-Tensorflow2-main\\Mask-R-CNN-using-Tensorflow2-main\\Airport-Trolley-aeroportalgerV1", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("C:\\Users\\yasmi\\Documents\\detection\\Mask-R-CNN-using-Tensorflow2-main\\Mask-R-CNN-using-Tensorflow2-main\\Airport-Trolley-aeroportalgerV1", "valid")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')
			
				
				
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)			