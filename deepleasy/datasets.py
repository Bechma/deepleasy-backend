import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.keras.datasets import mnist, cifar10, cifar100, imdb, reuters, fashion_mnist, boston_housing
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import to_categorical


class Dataset:
	def __init__(self, x_train, x_test, y_train, y_test, input_shape, output_shape, name):
		super().__init__()
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.name = name

	def toJSON(self):
		return {
			"name": self.name,
			"input_shape": self.input_shape,
			"output_shape": self.output_shape
		}

	def prepare_for_predict(self, x_train):
		pass

	def __str__(self):
		return self.name


class Cifar10(Dataset):
	def __init__(self):
		num_classes = 10

		# The data, split between train and test sets:
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()

		x_train = x_train.astype('float32') / 255
		x_test = x_test.astype('float32') / 255

		# Convert class vectors to binary class matrices.
		y_train = to_categorical(y_train, num_classes)
		y_test = to_categorical(y_test, num_classes)

		super().__init__(
			x_train,
			x_test,
			y_train,
			y_test,
			x_train.shape[1:],
			num_classes,
			'cifar10'
		)


class Cifar100(Dataset):
	def __init__(self):
		num_classes = 100

		# The data, split between train and test sets:
		(x_train, y_train), (x_test, y_test) = cifar100.load_data()

		x_train = x_train.astype('float32') / 255
		x_test = x_test.astype('float32') / 255

		# Convert class vectors to binary class matrices.
		y_train = to_categorical(y_train, num_classes)
		y_test = to_categorical(y_test, num_classes)

		super().__init__(
			x_train,
			x_test,
			y_train,
			y_test,
			x_train.shape[1:],
			num_classes,
			'cifar100'
		)


class IMDB(Dataset):
	def __init__(self):
		num_classes = 1
		self.max_features = 5000
		self.maxlen = 400

		(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)

		x_train = pad_sequences(x_train, maxlen=self.maxlen)
		x_test = pad_sequences(x_test, maxlen=self.maxlen)

		super().__init__(
			x_train,
			x_test,
			y_train,
			y_test,
			(self.maxlen,),
			num_classes,
			'imdb'
		)

	def prepare_for_predict(self, x_train):
		return pad_sequences(x_train, maxlen=self.maxlen)


class Reuters(Dataset):
	def __init__(self):
		max_words = 1000

		# The data, split between train and test sets:
		(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)
		num_classes = np.max(y_train) + 1
		self.tokenizer = Tokenizer(num_words=max_words)
		x_train = self.tokenizer.sequences_to_matrix(x_train, mode='binary')
		x_test = self.tokenizer.sequences_to_matrix(x_test, mode='binary')
		y_train = to_categorical(y_train, num_classes)
		y_test = to_categorical(y_test, num_classes)

		super().__init__(
			x_train,
			x_test,
			y_train,
			y_test,
			(max_words,),
			num_classes,
			'reuters'
		)

	def prepare_for_predict(self, x_train):
		return self.tokenizer.sequences_to_matrix(x_train, mode='binary')


class Mnist(Dataset):
	def __init__(self):
		num_classes = 10
		img_rows, img_cols = 28, 28

		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		if K.image_data_format() == 'channels_first':
			x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
			x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
			input_shape = (1, img_rows, img_cols)
		else:
			x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
			x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
			input_shape = (img_rows, img_cols, 1)

		x_train = x_train.astype('float32') / 255
		x_test = x_test.astype('float32') / 255
		y_train = to_categorical(y_train, num_classes)
		y_test = to_categorical(y_test, num_classes)

		super().__init__(
			x_train,
			x_test,
			y_train,
			y_test,
			input_shape,
			num_classes,
			'mnist'
		)


class FashionMnist(Dataset):
	def __init__(self):
		num_classes = 10
		img_rows, img_cols = 28, 28

		(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

		if K.image_data_format() == 'channels_first':
			x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
			x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
			input_shape = (1, img_rows, img_cols)
		else:
			x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
			x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
			input_shape = (img_rows, img_cols, 1)

		x_train = x_train.astype('float32') / 255
		x_test = x_test.astype('float32') / 255
		y_train = to_categorical(y_train, num_classes)
		y_test = to_categorical(y_test, num_classes)

		super().__init__(
			x_train,
			x_test,
			y_train,
			y_test,
			input_shape,
			num_classes,
			'fashion_mnist'
		)


class BostonHousing(Dataset):
	def __init__(self):
		(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

		mean = x_train.mean(axis=0)
		x_train -= mean
		std = x_train.std(axis=0)
		x_train /= std

		x_test -= mean
		x_test /= std

		super().__init__(
			x_train,
			x_test,
			y_train,
			y_test,
			(x_train.shape[1:]),
			1,
			'boston_housing'
		)

	def prepare_for_predict(self, x_train: np.ndarray):
		mean = x_train.mean(axis=0)
		x_train -= mean
		std = x_train.std(axis=0)
		x_train /= std
		return x_train

if __name__ == "__main__":
	Cifar10()
	Cifar100()
	IMDB()
	Reuters()
	Mnist()
	FashionMnist()
	BostonHousing