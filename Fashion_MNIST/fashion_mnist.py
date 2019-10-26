#  Fashion MNIST dataset class

import matplotlib.pyplot as plt
import math
import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

OUT_DIR = 'images/'


class FashionMNIST:

    def __init__(self, show=False, val_split=0.2, **kwargs):
        """Initialize Class FashionMNIST and load dataset Fashion MNIST dataset.

            Args:
                show (bool): True if we want to show some training images from Fashion MNIST.
                val_split (float): validation split to use for training.
                kwargs (dict): additional arguments to pass for data augmentation of the input images:
                    - rotation (int): rotation range for random rotations.
                    - shift (float): shift range for random height/width shifts.
                    - brightness (tuple): brightness range for random brightness shifts.
                    - flip (bool): whether to apply random horizontal flips.

            Returns:
                None.
            """

        print('\nLoading Fashion MNIST Dataset')

        # Load Fashion MNIST Dataset
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        print(x_train.shape)

        # Classes labels
        self.num_classes = 10
        self.classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        print('- training images shape : {}'.format(x_train.shape))
        print('- training labels shape : {}'.format(y_train.shape))
        print('- test images shape : {}'.format(x_test.shape))
        print('- test label shape : {}'.format(y_test.shape))

        # Convert inputs from uint8 to float32 and add one dimension at the end for the color channel
        self.x_train = np.expand_dims(x_train.astype(np.float32), axis=3)
        self.x_test = np.expand_dims(x_test.astype(np.float32), axis=3)

        # One-hot encode labels
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)

        # Define training and test generators for pixels rescaling in the interval [0, 1]
        # and setting the desired validation set and data augmentation
        self.validation_split = val_split
        self.train_generator_args = dict(rescale=1/255.,
                                         validation_split=val_split,
                                         rotation_range=kwargs['rotation'],
                                         width_shift_range=kwargs['shift'],
                                         height_shift_range=kwargs['shift'],
                                         brightness_range=kwargs['brightness'],
                                         horizontal_flip=kwargs['flip'],
                                         fill_mode='reflect')

        self.train_generator = ImageDataGenerator(**self.train_generator_args)
        self.test_generator = ImageDataGenerator(rescale=1/255.)  # the test generator exclusively rescale images

        # Eventually show some training images from Fashion MNIST
        if show:
            self._show_images_(x_train, y_train)

    def get_generator(self, batch_size=64, seed=None, subset='training'):
        """Get training/validation/test generator for generating batches of rescaled (and eventually augmented) data.

            Args:
                batch_size (int): size of the batches to be generated.
                seed (int): random seed.
                subset (string): 'training', 'validation' or 'test', the subset to generate from.

            Returns:
                generator (iterator): training/validation/test generator.
                steps_per_epoch (int): steps per epoch to run the generator for in order to provide the full subset.
            """

        if subset == 'test':
            # If test set, generate batches of (image, label) tuples from the test set
            generator = self.test_generator.flow(self.x_test, self.y_test,
                                                 batch_size=batch_size, shuffle=True,
                                                 seed=seed, subset=None)

        elif self.validation_split == 0. and subset == 'training':
            # If training set and no validation, generate batches of (image, label) tuples from the training set
            generator = self.train_generator.flow(self.x_train, self.y_train,
                                                  batch_size=batch_size, shuffle=True,
                                                  seed=seed, subset=None)

        elif self.validation_split == 0. and subset == 'validation':
            # If validation, but no validation is set, return a null generator
            generator = None
        else:
            # If training/validation, when validation is set, generate batches of (image, label)
            # tuples from the training/validation set
            generator = self.train_generator.flow(self.x_train, self.y_train,
                                                  batch_size=batch_size, shuffle=True,
                                                  seed=seed, subset=subset)

        # If the generator is not None, steps_per_epoch is computed and rounded above
        if generator:
            steps_per_epoch = math.ceil(generator.n / float(batch_size))
        else:
            steps_per_epoch = None

        return generator, steps_per_epoch

    def set_data_augmentation(self, **kwargs):
        """Set data augmentation.

            Args:
                kwargs (dict): arguments to set for data augmentation of the input images:
                    - rotation (int): rotation range for random rotations.
                    - shift (float): shift range for random height/width shifts.
                    - brightness (int): brightness range for random brightness shifts.
                    - flip (bool): whether to apply random horizontal flips.

            Returns:
                None.
            """

        self.train_generator_args['rotation_range'] = kwargs['rotation']
        self.train_generator_args['width_shift_range'] = kwargs['shift']
        self.train_generator_args['height_shift_range'] = kwargs['shift']
        self.train_generator_args['brightness_range'] = kwargs['brightness']
        self.train_generator_args['horizontal_flip'] = kwargs['flip']
        self.train_generator = ImageDataGenerator(**self.train_generator_args)

    def set_validation_split(self, val_split):
        """Set validation split.

            Args:
                val_split (float): validation split to use for training.

            Returns:
                None.
            """
        self.validation_split = val_split
        self.train_generator_args['validation_split'] = val_split
        self.train_generator = ImageDataGenerator(**self.train_generator_args)

    def _show_images_(self, x_train, y_train):
        """Show some training images from Fashion MNIST.

            Args:
                x_train (ndarray): training images to show.
                y_train (ndarray): training labels to show.

            Returns:
                None.
            """

        plt.figure(figsize=(7, 7))

        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_train[i], cmap=plt.cm.binary)
            plt.xlabel(self.classes[y_train[i]])

        plt.show()
