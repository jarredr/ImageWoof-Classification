import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
import keras
from keras.layers import Dropout, Flatten, Convolution2D, ZeroPadding2D, MaxPooling2D, Dense
from keras.models import Sequential
from keras.preprocessing import image
from PIL import Image


imagenet_classes = {
        'n02115641': 'Dingo',
        'n02111889': 'Samoyed',
        'n02105641': 'Old English Sheepdog',
        'n02099601': 'Golden Retriever',
        'n02096294': 'Australian Terrier',
        'n02093754': 'Border Terrier',
        'n02089973': 'English Foxhound',
        'n02088364': 'Beagle',
        'n02087394': 'Rhodesian Ridgeback',
        'n02086240': 'Shih-Tzu'
    }

epochs = 10
num_classes = 4
img_dim = 48


def data_loader(base_path, for_sklearn=False):
    """Loads the train and test sets of the Imagewoof dataset

    Args:
        base_path: A string providing the base path for the dataset
        for_sklearn: A boolean for whether to format the data for sklearn models

    Returns:
        Numpy arrays for the train and test sets plus the classes of the label encoder
    """

    def flatten_data(data):
        """Flattens the dimensions of the data for sklearn models

        Args:
            data: A multi-dimensional numpy array of image pixels to be flattened

        Returns:
            Data reshaped into one dimension
        """
        samples, dim1, dim2, alpha = data.shape
        data = data.reshape(samples, dim1 * dim2 * alpha)
        return data

    def load_image(image_path):
        """Loads an image from file and scales it

        Args:
            image_path: The path of the image on disk

        Returns:
            A numpy array of pixel values representing the image
        """
        img = image.load_img(image_path, grayscale=False, target_size=(img_dim, img_dim))
        x = image.img_to_array(img, dtype='float32')
        return x

    def load_dataset(breed_paths):
        """Loads all the images in a directory and normalizes them

        Args:
            breed_paths: A list of paths to the directories of each dog breed

        Returns:
            A numpy array of images in the dataset and a list of labels of the images in x
        """
        x = []
        y = []
        for i in range(num_classes):
            breed = breed_paths[i]
            breed_class = breed[-9:]
            image_paths = glob.glob(breed + '/*')
            for image_path in image_paths:
                x.append(load_image(image_path))
                y.append(breed_class)
        x = np.array(x)
        x /= 255
        return x, y

    print('Loading data...')
    train_paths = glob.glob(base_path + '/train/*')
    test_paths = glob.glob(base_path + '/val/*')
    x_train, y_train = load_dataset(train_paths)
    x_test, y_test = load_dataset(test_paths)
    if for_sklearn:
        x_train = flatten_data(x_train)
        x_test = flatten_data(x_test)
        encoder = LabelEncoder()
    else:
        encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    print('Finished loading data')
    return x_train, y_train, x_test, y_test, encoder.classes_


def get_class(prediction, classes):
    """Gets the class name of a prediction

    Args:
        prediction: A one-hot numpy array
        classes: A numpy array matching one-hot to class name

    Returns:
        The English name of the prediction
    """
    return imagenet_classes[classes[np.argmax(prediction)]]


def class_distribution(labels, classes):
    """Prints the distribution of classes in a dataset

    Args:
        labels: the true labels for the data
        classes: the classes from the encoder

    Returns:
        Nothing
    """
    label_counts = {}
    for label in labels:
        class_name = get_class(label, classes)
        if class_name in label_counts:
            label_counts[class_name] += 1
        else:
            label_counts[class_name] = 1

    for dog_breed, count in enumerate(label_counts):
        print(f'{dog_breed}: {count}')


def evaluate_model(model, x_train, y_train, x_test, y_test):
    """Trains a model and prints the train and test accuracy.

    Args:
        model: an sklearn model to be trained and evaluated
        x_train: train set of images
        y_train: labels of train set
        x_test: test set of images
        y_test: labels of test set

    Returns:
        The trained model
    """
    print('Training model...')
    model.fit(x_train, y_train)
    print('Evaluating model...')
    print(f'Train accuracy: {model.score(x_train, y_train)}')
    print(f'Test accuracy: {model.score(x_test, y_test)}')
    return model


def cnn(x_train, y_train, x_test, y_test):
    cnn_model = Sequential()
    cnn_model.add(ZeroPadding2D((1, 1), input_shape=(img_dim, img_dim, 3)))
    cnn_model.add(Convolution2D(32, (3, 3), activation='relu'))
    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Convolution2D(32, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Convolution2D(64, (3, 3), activation='relu'))
    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Convolution2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(Dropout(0.5))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(num_classes, activation='softmax'))

    cnn_model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy']
                      )

    cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=256)
    score = cnn_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    # x_train, y_train, x_test, y_test, encoder_classes = data_loader('./data/imagewoof', for_sklearn=True)
    # logistic_regression(x_train, y_train, x_test, y_test)
    # x_train, y_train, x_test, y_test, encoder_classes = data_loader('./data/imagewoof')
    # class_distribution(y_train, encoder_classes)
    # cnn(x_train, y_train, x_test, y_test)
    sx_train, sy_train, sx_test, sy_test, encoder_classes = data_loader('./data/imagewoof', for_sklearn=True)

    print('Logistic Regression:')
    lr_model = LogisticRegression(max_iter=100)
    evaluate_model(lr_model, sx_train, sy_train, sx_test, sy_test)

    print('SVM:')
    svm_model = SGDClassifier()
    evaluate_model(svm_model, sx_train, sy_train, sx_test, sy_test)


if __name__ == '__main__':
    main()
