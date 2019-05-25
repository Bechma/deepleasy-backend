import tensorflow.keras.backend as K
from tensorflow.python.keras.datasets.mnist import load_data
from tensorflow.python.keras.utils import to_categorical


def get_mnist():
	num_classes = 10
	img_rows, img_cols = 28, 28

	(x_train, y_train), (x_test, y_test) = load_data()

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)

	return x_train, x_test, y_train, y_test, input_shape
