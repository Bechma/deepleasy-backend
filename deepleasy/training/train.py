import math
import os
import time
from functools import reduce

import numpy as np
from celery import shared_task
from django.contrib.auth.models import User
from sklearn.utils.linear_assignment_ import linear_assignment
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.python.keras.models import Model

from deepleasy.control.options import DATASETS
from deepleasy.models import Progress, History
from deepleasy.training.clustering import train_autoencoder, build_clustering_model


class LossHistory(Callback):
	def __init__(self, prog):
		super().__init__()
		self.losses = []
		self.prog = prog

	def on_train_begin(self, logs=None):
		self.losses = []

	def on_epoch_end(self, epoch, logs=None):
		self.losses.append({"accuracy": str(logs.get('acc')), "loss": str(logs.get("loss"))})

	def on_epoch_begin(self, epoch, logs=None):
		self.prog.epochs = epoch
		self.prog.save()


def get_training_data(dataset: str):
	for i in DATASETS:
		if i.name == dataset:
			return i.x_train, i.x_test, i.y_train, i.y_test, i.input_shape
	raise ValueError


@shared_task
def build_model_supervised(model_info: dict, username: str, model_path: str):
	print(model_info)

	user = User.objects.get(username=username)
	prog = Progress.objects.get(user=user)
	prog.task_id = build_model_supervised.request.id
	prog.save()
	history = History(timestamp=int(time.time()), user=user, path="", steps_info={}, dataset=model_info["dataset"])

	x_train, x_test, y_train, y_test, input_shape = get_training_data(model_info["dataset"])

	last_layer = Input(shape=input_shape)
	first_layer = last_layer

	for layer in model_info["layers"]:
		if layer["name"] == "Dense":
			last_layer = Dense(layer["units"], layer["activation"])(last_layer)
		elif layer["name"] == "Dropout":
			last_layer = Dropout(layer["rate"])(last_layer)
		elif layer["name"] == "Conv2D":
			last_layer = Conv2D(layer["units"], (layer["kernel"]["x"], layer["kernel"]["y"]), activation=layer["activation"])(last_layer)
		elif layer["name"] == "MaxPooling2D":
			last_layer = MaxPooling2D((layer["kernel"]["x"], layer["kernel"]["y"]), padding=layer["padding"])(last_layer)
		elif layer["name"] == "Flatten":
			last_layer = Flatten()(last_layer)

	model = Model(inputs=first_layer, outputs=last_layer)
	model.compile(optimizer=model_info["optimizer"], loss=model_info["loss"], metrics=["accuracy"])

	prog.running = True
	prog.save()

	callback = LossHistory(prog)
	try:
		model.fit(x_train, y_train, batch_size=int(model_info["batchSize"]), epochs=int(model_info["epochs"]),
				  callbacks=[callback], verbose=2, validation_split=1.0 - model_info["trainPercentage"])
	except KeyboardInterrupt:
		history.steps_info = {"message": "Task cancelled by admin"}
		history.save()
		prog.delete()
		return
	except InvalidArgumentError:
		history.steps_info = {"message": "Dimensions error, probably last layer"}
		history.save()
		prog.delete()
		return
	except:
		history.steps_info = {"message": "There was an error whilst training"}
		history.save()
		prog.delete()
		return

	accuracy = model.evaluate(x_test, y_test)

	path = os.path.join(model_path, "{}_{}.h5".format(username, int(time.time())))
	model.save(path)

	# saved_model_path = save_keras_model(model, model_path, as_text=True)
	prog.delete()

	history.path = path  # saved_model_path.decode()
	history.accuracy = accuracy[1] if not math.isnan(accuracy[1]) else 0.0
	history.loss = accuracy[0] if not math.isnan(accuracy[0]) else 0.0
	history.steps_info = callback.losses
	history.save()


@shared_task
def build_model_unsupervised(model_info: dict, username, model_path: str):
	print(model_info)

	x_train, x_test, y_train, y_test, input_shape = get_training_data(model_info["dataset"])

	one_dimension = reduce(lambda x, y: x * y, input_shape)
	x_train = x_train.reshape(-1, one_dimension)

	user = User.objects.get(username=username)
	prog = Progress.objects.get(user=user)
	prog.task_id = build_model_unsupervised.request.id
	prog.save()
	history = History(timestamp=int(time.time()), user=user, path="", steps_info={}, dataset=model_info["dataset"])

	n_features = np.shape(x_train)[1]
	model_info["input4encoder"] = n_features

	encoder = train_autoencoder(x_train, model_info)
	model = build_clustering_model(x_train, encoder, model_info["n_clusters"])
	steps = []
	try:
		y_train = y_train.argmax(1)
	except:
		pass

	for i in range(model_info["epochs"]):  # walk through iterations
		predictions = model.predict(x_train)  # predict clusters
		weights = predictions ** 2 / predictions.sum(0)  # calculate
		targets = (weights.T / weights.sum(1)).T  # target distribution
		loss = model.train_on_batch(x=x_train, y=targets)  # update model weights

		y_pred = predictions.argmax(1)
		D = max(y_pred.max(), y_train.max()) + 1
		confusion_matrix = np.zeros((D, D), dtype=np.int64)
		# Confusion matrix.
		for j in range(y_pred.size):
			confusion_matrix[y_pred[j], y_train[j]] += 1
		ind = linear_assignment(-confusion_matrix)
		acc = sum([confusion_matrix[x, y] for x, y in ind]) * 1.0 / y_pred.size
		steps.append({"accuracy": str(acc), "loss": str(loss)})

	path = os.path.join(model_path, "{}_{}.h5".format(username, int(time.time())))
	model.save(path)

	history.path = path
	history.accuracy = acc
	history.loss = loss
	history.steps_info = steps
	history.save()

	prog.delete()

