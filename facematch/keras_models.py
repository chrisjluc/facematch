import consts

def keras_auto_encoder(input_size, encoding_size):
    from keras.models import Model
    from keras.layers import Dense, Input
    from keras import regularizers

    _input = Input(shape=(input_size,))
    encoded = Dense(
            encoding_size,
            activation='relu',
            activity_regularizer=regularizers.activity_l1(consts.sae_regularizer)
            )(_input)
    decoded = Dense(input_size, activation='sigmoid')(encoded)

    autoencoder = Model(input=_input, output=decoded)
    encoder = Model(input=_input, output=encoded)

    return autoencoder, encoder


def keras_cnn_nn1(nb_classes, input_shape):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D((1,1)))
    model.add(AveragePooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def keras_cnn_nn2(nb_classes, input_shape):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=input_shape))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(ZeroPadding2D((1,1)))
    model.add(AveragePooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

