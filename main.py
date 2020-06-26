import glob
import argparse
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
import keras
from keras.layers import Dropout, Flatten, Convolution2D, ZeroPadding2D, MaxPooling2D, Dense, GlobalAveragePooling2D, \
    Activation
from keras.models import Sequential, Model
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


def data_loader(base_path, for_sklearn, img_dim, num_classes):
    """Loads the train and test sets of the Imagewoof dataset

    Args:
        base_path: A string providing the base path for the dataset
        for_sklearn: A boolean for whether to format the data for sklearn models
        img_dim: Number of dimensions in the image
        num_classes: The number of classes (breeds) to use
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


def cnn(x_train, y_train, x_test, y_test, epochs, img_dim, num_classes):
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


def large_cnn(x_train, y_train, x_test, y_test, epochs, img_dim, num_classes):
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

    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Convolution2D(128, (3, 3), activation='relu'))
    cnn_model.add(ZeroPadding2D((1, 1)))
    cnn_model.add(Convolution2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(Dropout(0.5))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dropout(0.5))
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


def vgg_face(x_train, y_train, x_test, y_test, epochs, num_classes):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))
    # By default, layers beyond this are trainable
    model.add(Convolution2D(4096, (7, 7), activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1), name='fc8'))
    model.add(Flatten())
    model.add(Activation('softmax'))

    # pre-trained weights of vgg-face model.
    # you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
    # related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
    model.load_weights('data/vgg_face_weights.h5')

    for layer in model.layers[:-7]:
        layer.trainable = False

    vgg_face_output = Sequential()
    vgg_face_output = Convolution2D(num_classes, (1, 1), name='predictions')(model.layers[-4].output)
    vgg_face_output = Flatten()(vgg_face_output)
    vgg_face_output = Activation('softmax')(vgg_face_output)

    vgg_face_model = Model(inputs=model.input, outputs=vgg_face_output)

    vgg_face_model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(),
                           # optimizer = sgd,
                           metrics=['accuracy']
                           )

    vgg_face_model.fit(x_train, y_train, epochs=epochs, batch_size=32)
    model.save_weights('./data/vgg_face_checkpoint')
    score = vgg_face_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def resnet50(x_train, y_train, x_test, y_test, epochs, img_dim, num_classes):
    frozen_to = -1
    input_shape = (img_dim, img_dim, 3)
    original_model = keras.applications.ResNet50V2(weights='imagenet', input_shape=input_shape, include_top=False)
    input_layer = original_model.input
    intermediate_layer = GlobalAveragePooling2D()
    intermediate_layer = intermediate_layer(original_model.layers[-1].output)
    prediction_layer = Dense(num_classes, activation='softmax')
    output_layer = prediction_layer(intermediate_layer)
    model = Model(input_layer, output_layer)
    for _, layer in enumerate(model.layers[:frozen_to]):
        layer.trainable = False
    for _, layer in enumerate(model.layers[frozen_to:]):
        layer.trainable = True
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=epochs)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    # print('Logistic Regression:')
    # lr_model = LogisticRegression(max_iter=100)
    # evaluate_model(lr_model, sx_train, sy_train, sx_test, sy_test)
    # print('SVM:')
    # svm_model = SGDClassifier()
    # evaluate_model(svm_model, sx_train, sy_train, sx_test, sy_test)
    parser = argparse.ArgumentParser()
    parser.add_argument('data_type')
    parser.add_argument('model_type')
    parser.add_argument('model_name')
    parser.add_argument('epochs', type=int)
    parser.add_argument('classes', type=int)
    args = parser.parse_args()
    if args.data_type == 'small':
        img_dim = 48
        if args.model_type == 'keras':
            x_train, y_train, x_test, y_test, encoder_classes = data_loader('./data/imagewoof', for_sklearn=False,
                                                                            img_dim=img_dim, num_classes=args.classes)
        elif args.model_type == 'sklearn':
            x_train, y_train, x_test, y_test, encoder_classes = data_loader('./data/imagewoof', for_sklearn=True,
                                                                            img_dim=img_dim, num_classes=args.classes)
        else:
            raise ValueError('Invalid model type')
    elif args.data_type == 'large':
        img_dim = 224
        x_train, y_train, x_test, y_test, encoder_classes = data_loader('./data/imagewoof', for_sklearn=False,
                                                                        img_dim=img_dim, num_classes=args.classes)
    else:
        raise ValueError('Invalid data type')
    if args.model_name == 'cnn':
        cnn(x_train, y_train, x_test, y_test, epochs=args.epochs, img_dim=img_dim, num_classes=args.classes)
    elif args.model_name == 'large_cnn':
        large_cnn(x_train, y_train, x_test, y_test, epochs=args.epochs, img_dim=img_dim, num_classes=args.classes)
    elif args.model_name == 'vgg_face':
        if args.data_type == 'small':
            raise ValueError('Only large data is accepted in VGG_Face')
        vgg_face(x_train, y_train, x_test, y_test, epochs=args.epochs, num_classes=args.classes)
    elif args.model_name == 'resnet50':
        resnet50(x_train, y_train, x_test, y_test, epochs=args.epochs, img_dim=img_dim, num_classes=args.classes)
    else:
        raise ValueError('Invalid model name')


if __name__ == '__main__':
    main()
