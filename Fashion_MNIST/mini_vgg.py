# Mini VGG Class

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

OUT_DIR = 'images/'
CHECK_DIR = 'checkpoints/'


class MiniVGG:

    def __init__(self, conv_layers=4, input_shape=(28, 28, 1), load_best_weights=False, name='miniVGG', num_classes=10):
        """Initialize Class MiniVGG and build a VGG2 or VGG4 convolutional neural network model.

            Args:
                conv_layers (int): 2 or 4, the number of convolutional layers to use.
                input_shape (tuple): input shape of the model, (28, 28, 1) per default as in MNIST.
                load_best_weights (bool): True if model best weights have to be loaded from the .hdf5 file.
                name (string): name of the model.
                num_classes (int): number of classes, 10 per default as in MNIST.

            Returns:
                None.
            """

        # Define the model name, the path where to save the best weights,
        # the number of convolutional layers, the input shape and the number of classes
        self.name = name
        self.best_weights_path = CHECK_DIR + name + 'weights_best.hdf5'
        self.conv_layers = conv_layers
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Build the model and compile it
        self._build_model_()
        self.compile_model()

        # Eventually load weights of the best model found
        if load_best_weights:
            self.model.load_weights(self.best_weights_path)

    def _build_model_(self):
        """Build a VGG2 or VGG4 convolutional neural network model.

            Args:
                None.

            Returns:
                None.
            """

        print('\nBuild Model')

        # Build Sequential model and add 2x(Conv + BN) + MaxPool + Drop
        # Filters are set to be (3, 3), ReLU activations and same padding as in VGG16
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # Eventually add another 2x(Conv + BN) + MaxPool + Drop with doubled depth
        if self.conv_layers == 4:
            self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

        # After the features selection, flat the features and add 2 Fully-Connected layers,
        # with BN and Dropout in the middle. The output layer outputs num_classes probabilities by softmax.
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Summary and plot of the model
        self.model.summary()
        plot_model(self.model, to_file=OUT_DIR + self.name + '_model.png', show_shapes=True, show_layer_names=True)

    def compile_model(self, init_lr=1e-2):
        """Compile the MiniVGG model.

            Args:
                init_lr (float): initial learning_rate.

            Returns:
                None.
            """
        # The Adam optimizer is used, with categorical cross_entropy as loss and accuracy as evaluation metric
        self.model.compile(optimizer=Adam(lr=init_lr),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def load_best_weights(self):
        """Load MiniVGG model best weights.

           Args:
               None.

           Returns:
               None.
           """
        self.model.load_weights(self.best_weights_path)

    def _plot_helper_(self, x_axis, history, plot_type, ylabel):
        """Plot helper.

            Args:
                x_axis (ndarray): x axis array to plot versus.
                history (history object): training history to plot from.
                plot_type (string): 'loss' or 'accuracy'.
                ylabel (string): y axis label.

            Returns:
                None.
            """

        # Create new figure
        plt.figure()

        # Plot training and validation loss or accuracy
        plt.plot(x_axis, history.history[plot_type], label='Training')
        plt.plot(x_axis, history.history['val_{}'.format(plot_type)], label='Validation')

        # Set title, legend, labels and save figure
        plt.title('Fashion MNIST - {} vs. epochs'.format(ylabel))
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.savefig(OUT_DIR + '{}_{}.png'.format(self.name, plot_type))

    def plot_history(self, history):
        """Plot Training History.

            Args:
                history (history object): training history to plot from.

            Returns:
                None.
            """

        print('\nPlot History')
        # Set plot style
        plt.style.use("ggplot")

        # Create epochs array
        epochs = np.arange(0, len(history.history['loss']))

        # Plot loss and accuracy
        self._plot_helper_(epochs, history, plot_type='loss', ylabel='Cross-Entropy Loss')
        self._plot_helper_(epochs, history, plot_type='accuracy', ylabel='Accuracy')

    def predict(self, classes, input_image):
        """Predict a single image.

            Args:
                classes (list): classes names.
                input_image (ndarray): input image to predict.

            Returns:
                predicted_class (string): class predicted.
                prediction_probability (float): probability of the prediction.
            """
        print('\nPredict')

        # Make a copy of the input image
        image = np.copy(input_image)

        # Convert to grayscale if the input shape require so
        if self.input_shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to match the input size
        image = cv2.resize(image, dsize=(self.input_shape[0], self.input_shape[1]))

        # Rescale image and insert an empty dimension at the beginning for batch processing
        image = image.astype(np.float32) / 255.
        image = np.expand_dims(image, axis=0)

        # Predict and compute the most probable class
        predictions = self.model.predict(image)
        predicted_index = predictions.argmax()

        return classes[predicted_index], predictions[predicted_index]

    def test(self, dataset, batch_size=64):
        """Test the MiniVGG model.

            Args:
                dataset (dataset class): dataset to test from that implements a Keras ImageGenerator.
                batch_size (int): batch size.

            Returns:
                test_loss (float): loss on the test set.
                test_accuracy (float): accuracy on the test set.
            """

        print('\nTest')

        # Test the dataset with a generator that generate batches of rescaled data from the dataset's test set
        test_generator, test_steps = dataset.get_generator(batch_size=batch_size, subset='test')
        test_loss, test_accuracy = self.model.evaluate_generator(generator=test_generator,
                                                                 steps=test_steps,
                                                                 verbose=1)

        # Generate classification report
        predictions = self.model.predict(dataset.x_test / 255.)

        print(classification_report(dataset.y_test.argmax(axis=1),
                                    predictions.argmax(axis=1),
                                    target_names=dataset.classes,
                                    digits=4))

        return test_loss, test_accuracy

    def train(self, dataset, epochs=50, batch_size=64):
        """Train the MiniVGG model.

            Args:
                dataset (dataset class): dataset to test from that implements a Keras ImageGenerator.
                epochs (int): epochs to run.
                batch_size (int): batch size.


            Returns:
                train_history (history object): training history.
            """

        print('\nTrain')

        # If a validation set is used, we can use ModelCheckpoint, ReduceLROnPlateau and eventually EarlyStopping
        if dataset.validation_split > 0.:

            # ModelCheckpoint is used to saved the weights of the model that report the best validation loss
            checkpoint = ModelCheckpoint(self.best_weights_path, monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=True, mode='min')

            # ReduceLROnPlateau is used to reduce the learning rate if a plateau in the validation loss is reached
            # This helps with slowing down the optimization search when approaching the a global/local minimum
            reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                                     mode='min', min_delta=1e-4, min_lr=1e-8)

            # EarlyStopping stops training when the validation loss does not improve for a while
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=6,
                                           verbose=1, mode='min', restore_best_weights=True)

            callbacks = [checkpoint, reduce_lr_on_plateau, early_stopping]  # list of callbacks to apply during training

        else:
            callbacks = None  # else, use no callbacks

        # Train the dataset with a generator that generate batches of rescaled (and eventually augmented) data
        # from the dataset's training/validation set
        train_generator, train_steps = dataset.get_generator(batch_size=batch_size, subset='training')
        val_generator, val_steps = dataset.get_generator(batch_size=batch_size, subset='validation')

        train_history = self.model.fit_generator(generator=train_generator,
                                                 steps_per_epoch=train_steps,
                                                 validation_data=val_generator,
                                                 validation_steps=val_steps,
                                                 callbacks=callbacks, epochs=epochs, verbose=2,
                                                 workers=10, use_multiprocessing=True)

        # Plot the training history
        self.plot_history(train_history)

        return train_history
