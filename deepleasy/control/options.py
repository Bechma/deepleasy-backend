import numpy as np

from deepleasy.datasets import Cifar10, Cifar100, IMDB, Reuters, Mnist, FashionMnist, BostonHousing

OPTIMIZERS = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop', 'SGD']

LOSSES = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_hinge', 'cosine', 'cosine_proximity', 'hinge',
		  'kullback_leibler_divergence', 'logcosh', 'mean_absolute_error', 'mean_absolute_percentage_error',
		  'mean_squared_error', 'mean_squared_logarithmic_error', 'poisson', 'sparse_categorical_crossentropy',
		  'squared_hinge']

old = np.load
np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)
DATASETS = [
	Cifar10(),
	Cifar100(),
	IMDB(),
	Reuters(),
	Mnist(),
	FashionMnist(),
	BostonHousing()
]
np.load = old

LAYERS = ["Dense", "Conv2D", "Dropout", "MaxPooling2D", "Flatten"]

ACTIVATIONS = [
	"softmax",
	"elu",
	"selu",
	"softplus",
	"softsign",
	"relu",
	"tanh",
	"sigmoid",
	"hard_sigmoid",
	"exponential",
	"linear"
]
