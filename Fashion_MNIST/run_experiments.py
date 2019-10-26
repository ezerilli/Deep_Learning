from fashion_mnist import FashionMNIST
from mini_vgg import MiniVGG

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

VID_DIR = 'videos/'  # output directories
OUT_DIR = 'images/'
CHECK_DIR = 'checkpoints/'

BATCH_SIZE = 64  # sizes of the batches to compute
EPOCHS = 50  # number of epochs to run


def experiment1(load_weights=True):
    """Perform experiment 1.

        Args:
            load_weights (bool): True, if best weights have to be loaded, train otherwise.
        Returns:
            None.
        """

    # Import Fashion MNIST datasert with no data augmentation and with a 10% validation set
    fashion_mnist = FashionMNIST(show=False, val_split=0.1, rotation=0, shift=0., brightness=None, flip=False)

    # Dataset generators that generate batches of rescaled data from the dataset's training, validation and test set
    train_generator, train_steps = fashion_mnist.get_generator(batch_size=BATCH_SIZE, subset='training')
    val_generator, val_steps = fashion_mnist.get_generator(batch_size=BATCH_SIZE, subset='validation')
    test_generator, test_steps = fashion_mnist.get_generator(batch_size=BATCH_SIZE, subset='test')

    # Train and test a MiniVGG4 model
    print('\n\n')
    print('-' * 50)
    print('MiniVGG4')
    minivgg4 = MiniVGG(conv_layers=2, input_shape=fashion_mnist.x_train.shape[1:3], load_best_weights=load_weights,
                       name='minivgg4', num_classes=fashion_mnist.num_classes)

    # If best weights are not loaded, train
    if not load_weights:
        minivgg4.train(train_generator, val_generator, train_steps=train_steps, val_steps=val_steps,
                       validation_split=fashion_mnist.validation_split, epochs=EPOCHS)

    minivgg4.test(test_generator, test_steps=test_steps)
    minivgg4.test_report(fashion_mnist.x_test, fashion_mnist.y_test, classes=fashion_mnist.classes)

    # Train and test a MiniVGG6 model
    print('\n\n')
    print('-' * 50)
    print('MiniVGG6')
    minivgg6 = MiniVGG(conv_layers=4, input_shape=fashion_mnist.x_train.shape[1:3], load_best_weights=load_weights,
                       name='minivgg6', num_classes=fashion_mnist.num_classes)

    # If best weights are not loaded, train
    if not load_weights:
        minivgg6.train(train_generator, val_generator, train_steps=train_steps, val_steps=val_steps,
                       validation_split=fashion_mnist.validation_split, epochs=EPOCHS)

    minivgg6.test(test_generator, test_steps=test_steps)
    minivgg6.test_report(fashion_mnist.x_test, fashion_mnist.y_test, classes=fashion_mnist.classes)

    # Train and test a MiniVGG6 model with data augmentation (horizontal flips only)
    print('\n\n')
    print('-' * 50)
    print('MiniVGG6 with Data Augmentation')
    fashion_mnist.set_data_augmentation(rotation=0, shift=0., brightness=None, flip=True)
    minivgg6_augmented = MiniVGG(conv_layers=4, input_shape=fashion_mnist.x_train.shape[1:3],
                                 load_best_weights=load_weights, name='minivgg6_augmented',
                                 num_classes=fashion_mnist.num_classes)

    # If best weights are not loaded, train
    if not load_weights:
        minivgg6_augmented.train(train_generator, val_generator, train_steps=train_steps, val_steps=val_steps,
                                 validation_split=fashion_mnist.validation_split, epochs=EPOCHS)

    minivgg6_augmented.test(test_generator, test_steps=test_steps)
    minivgg6_augmented.test_report(fashion_mnist.x_test, fashion_mnist.y_test, classes=fashion_mnist.classes)

    # Randomly select a few test images
    print('\nRandomly predict 15 test images')
    j = 0
    plt.figure(figsize=(20, 6))
    for i in np.random.choice(np.arange(0, len(fashion_mnist.x_test)), size=(15,)):

        # Predict and compute the most probable class
        predictions = minivgg6_augmented.predict(fashion_mnist.x_test[i])
        prediction = predictions.argmax()  # predicted class
        probability = predictions[prediction]  # probability associated
        true_label = int(np.argmax(fashion_mnist.y_test[i]))  # true class
        print('true label = {}, prediction = {}, probability = {:.2f}'.format(true_label, prediction, probability))

        # Plot image and predictions histogram
        plt.subplot(5, 6, 2 * j + 1)
        image = fashion_mnist.x_test[i, :, :, 0].astype('uint8')  # rescale and reformat image
        plot_image(image, true_label, prediction, probability, classes=fashion_mnist.classes)
        plt.subplot(5, 6, 2 * j + 2)
        plot_value_array(true_label, prediction, predictions)
        j += 1

    plt.tight_layout()
    plt.savefig(OUT_DIR + 'test_image.png')


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video.
        frame_size (tuple): Width, height tuple of output video.
        fps (int): Frames per second.
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Video writer in MP4
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def plot_image(image, true_label, prediction, probability, classes):
    """Plot image with true label and associated prediction.

        Args:
            image (ndarrray): Image to plot.
            true_label (int): True label.
            prediction (int): Predicted label.
            probability (float): Probability of the predicted label.
            classes (list): List of classes names.

        Returns:
            None.
        """

    # Initialize the text label color as green (correct)
    text_color = 'green'
    # Otherwise, the class label prediction is incorrect
    if prediction != true_label:
        text_color = 'red'

    # Plot image with true label and associated prediction
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    label = plt.xlabel("{} {:2.0f}% ({})".format(classes[prediction], 100 * probability, classes[true_label]))
    label.set_color(text_color)


def plot_value_array(true_label, prediction, predictions):
    """Plot histogram of predictions.

        Args:
            true_label (int): True label.
            prediction (int): Predicted label.
            predictions (float): Probabilities of the prediction.

        Returns:
            None.
        """

    # Plot histogram of predictions
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    histogram = plt.bar(range(10), predictions, color="#777777")
    plt.ylim([0, 1])

    histogram[prediction].set_color('red')
    histogram[true_label].set_color('green')


def process_video(video_file, cnn):
    """Process a video for object detection and recognition.

        Args:
            video_file (string): Filename for saved video.
            cnn (cnn class): Convolutional Neural Network to use for recognition.
        Returns:
            None.
        """
    print("\nProcessing video:")

    fps = 30  # frames per second rate
    frames_to_save = [0, 80, 500, 600, 900]  # list of frames to save as examples
    video = os.path.join(VID_DIR, video_file)  # video path

    # Starts a video frame generator to generate one frame at time, then generate 1st frame
    image_gen = video_frame_generator(video)
    image_frame = image_gen.__next__()
    h, w = image_frame.shape[:2]  # height, width of each frame

    # Get the output video path and the output video writer
    out_path = os.path.join(VID_DIR, 'out_' + video_file)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    frame_num, output_counter = 1, 1  # initialize counters

    # While there is a frame to process
    while image_frame is not None:

        # Process current frame for object detection and recognition
        print("Processing frame {}".format(frame_num))
        # image_frame = process_frame(image_frame, cnn)

        # Get current frame to save and, if it is the current frame, save the processed frame and increment counter
        frame_id = frames_to_save[(output_counter - 1) % 5]
        if frame_num == frame_id:
            cv2.imwrite(os.path.join(OUT_DIR, 'out_frame_{}.png'.format(output_counter)), image_frame)
            output_counter += 1

        # Write processed frame to video
        video_out.write(image_frame)

        # Get next image frame and increment counter
        image_frame = image_gen.__next__()
        frame_num += 1

    # Release video writer
    video_out.release()


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()  # read the video

        if ret:
            yield frame  # return just (yield) current frame
        else:
            break

    video.release()  # Release Video Capture
    yield None


def process_frame(frame, cnn):

    # TODO docstring, comments, adapt to Fashion MNIST, test

    frame_resized = cv2.resize(frame, dsize=(1280, 720))
    frame_blurred = cv2.GaussianBlur(frame_resized, (5, 5), 3)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    frame_edges = cv2.Canny(frame_gray, 200, 220)
    # cv2.imshow('image', frame_edges)
    # cv2.waitKey(0)
    frame_mask = cv2.dilate(frame_edges, np.ones((5, 5), np.uint8), iterations=10)
    frame_masked = cv2.bitwise_and(frame_resized, frame_resized, mask=frame_mask)
    # cv2.imshow('image', frame_masked)
    # cv2.waitKey(0)

    window_sizes = [32 * i for i in range(1, frame_resized.shape[0] // 32)]
    candidates, scores = [], []
    thresh = 0.9
    for size in tqdm(window_sizes):
        # print('size = {}'.format(size))
        x_steps, y_steps = (frame_resized.shape[1] - size) // 32, (frame_resized.shape[0]-size) // 32

        for j in range(y_steps):
            for i in range(x_steps):
                x, y = 32 * i, 32 * j
                x2, y2 = x + size, y + size

                frame_roi = frame_resized[y:y2, x:x2]
                frame_masked_roi = frame_masked[y:y2, x:x2]
                mae = (np.absolute(frame_roi - frame_masked_roi)).mean(axis=None)

                if mae < 50.0:

                    (n, pn), (l1, pl1), (l2, pl2), (l3, pl3), (l4, pl4) = cnn.predict(frame_roi)

                    # print("n: {} ({:.2f})".format(n, pn))
                    # print("l1: {} ({:.2f})".format(l1, pl1))
                    # print("l2: {} ({:.2f})".format(l2, pl2))
                    # print("l3: {} ({:.2f})".format(l3, pl3))
                    # print("l4: {} ({:.2f})".format(l4, pl4))

                    if n == '1' and pn > thresh and pl1 > thresh:
                        label = '{}'.format(l1)
                        prob = pn * pl1
                    elif n == '2' and pn > thresh and pl1 > thresh and pl2 > thresh:
                        label = '{}{}'.format(l1, l2)
                        prob = pn * pl1 * pl2

                    elif n == '3' and pn > thresh and pl1 > thresh and pl2 > thresh and pl3 > thresh:
                        label = '{}{}{}'.format(l1, l2, l3)
                        prob = pn * pl1 * pl2 * pl3

                    elif n == '4' and pn > thresh and pl1 > thresh and pl2 > thresh and pl3 > thresh and pl4 > thresh:
                        label = '{}{}{}{}'.format(l1, l2, l3, l4)
                        prob = pn * pl1 * pl2 * pl3 * pl4
                    else:
                        label = 'Background'
                        prob = 1.0 - pn * pl1 * (1.0 - pl2 - pl2 * pl3 - pl2 * pl3 * pl4)

                    if label != 'Background':
                        candidates.append([x, y, x2, y2, int(label)])
                        scores.append(prob)

    candidates = np.array(candidates)
    scores = np.array(scores)
    # print('\ncandidates = ')
    # print(candidates)
    # print('\nscores = ')
    # print(scores)

    # if len(candidates) != 0:
    boxes = non_maximum_suppression(candidates, scores, 0.5)
    box = boxes[0]

    if len(boxes) != 0:
        cv2.rectangle(frame_resized, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        font_size = 5.0 * (int(box[2] - int(box[0])))/frame.shape[0]
        cv2.putText(frame_resized, '{}'.format(box[4]), (int(box[2]) + 10, int(box[3])), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (0, 0, 255), 2)
    # cv2.imshow('image', frame_resized)
    # cv2.waitKey(0)
    return frame_resized


def non_maximum_suppression(candidates, scores, iou_threshold):

    # TODO docstring, comments, test

    x1 = candidates[:, 0]
    y1 = candidates[:, 1]
    x2 = candidates[:, 2]
    y2 = candidates[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = scores.argsort().tolist()
    boxes = []

    while len(indices):
        index = indices.pop()
        boxes.append(index)
        if not len(indices):
            break
        ious = compute_iou(candidates[index], candidates[indices], area[index], area[indices])
        thresholded_indices = set((ious > iou_threshold).nonzero()[0])
        indices = [v for (i, v) in enumerate(indices) if i not in thresholded_indices]

    # print('boxes = {}'.format(np.array(candidates[boxes])))
    return np.array(candidates[boxes])


def compute_iou(box, boxes, box_area, boxes_area):

    # TODO docstring, comments, test

    max_x1 = np.maximum(box[0], boxes[:, 0])
    max_y1 = np.maximum(box[1], boxes[:, 1])
    min_x2 = np.minimum(box[2], boxes[:, 2])
    min_y2 = np.minimum(box[3], boxes[:, 3])

    intersections = np.maximum(min_y2 - max_y1, 0) * np.maximum(min_x2 - max_x1, 0)
    unions = box_area + boxes_area - intersections
    return intersections / unions


if __name__ == "__main__":

    experiment1(load_weights=True)  # change to False to do training

    # process_video(video_file="my_video.mp4", cnn=fashion_net)
