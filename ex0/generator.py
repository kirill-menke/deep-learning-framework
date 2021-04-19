import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
from math import sqrt, ceil
from numpy import random
from skimage.transform import resize


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False,
                 mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.batch_size = batch_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.file_index = 0

        with open(label_path) as f:
            labels = json.load(f)
        files = os.listdir(file_path)
        images = [resize(np.load(os.path.join(file_path, file)), image_size) for file in files]
        labels = [labels[os.path.splitext(file)[0]] for file in files]
        self.data = list(zip(images, labels))

        if shuffle:
            random.shuffle(self.data)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        num_files = len(self.data)
        batch = [self.data[(self.file_index + i) % num_files] for i in range(self.batch_size)]

        self.file_index += self.batch_size
        if self.file_index > num_files:
            self.file_index %= num_files
            if self.shuffle:
                random.shuffle(self.data)

        images, labels = zip(*batch)
        return np.array([self.augment(img) for img in images]), labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        if self.rotation:
            for _ in range(random.choice(4)):
                img = np.rot90(img)
        if self.mirroring:
            if random.choice(2):
                img = np.fliplr(img)

        return img

    @staticmethod
    def class_name(int_label):
        # This function returns the class name for a specific input

        return {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog',
                6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}[int_label]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.

        images, labels = self.next()
        n = int(ceil(sqrt(len(images))))
        _, ax = plt.subplots(nrows=n, ncols=n)
        for (i, l, s) in zip(images, labels, [col for row in ax for col in row]):
            s.imshow(i)
            s.set_title(self.class_name(l))
