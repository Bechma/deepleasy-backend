import numpy as np
from celery import shared_task
from django.contrib.auth.models import User
from tensorflow.contrib.saved_model import save_keras_model
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential

from deepleasy.dataset.mnist import get_mnist
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
	if dataset == "mnist":
		return get_mnist()


def init_prog_hist(user: str, dataset):
	user = User.objects.get(username=user)
	prog = Progress.objects.get(user=user)
	prog.task_id = build_model_supervised.request.id
	prog.save()
	history = History(user=user, path="", steps_info={}, dataset=dataset)
	return prog, history


@shared_task
def build_model_supervised(model_info: dict, user: str, model_path: str):
	print(model_info)

	model = Sequential()
	first = True

	prog, history = init_prog_hist(user, model_info["dataset"])

	x_train, x_test, y_train, y_test, input_shape = get_training_data(model_info["dataset"])

	for layer in model_info["layers"]:
		try:
			if layer["name"] == "Dense":
				if first:
					model.add(Dense(layer["units"], layer["activation"], input_shape=input_shape))
					first = False
				else:
					model.add(Dense(layer["units"], layer["activation"]))
			elif layer["name"] == "Dropout":
				model.add(Dropout(layer["rate"]))
			elif layer["name"] == "Conv2D":
				if first:
					first = False
					model.add(
						Conv2D(layer["units"], (layer["kernel"]["x"], layer["kernel"]["y"]),
							   activation=layer["activation"], input_shape=input_shape))
				else:
					model.add(
						Conv2D(layer["units"], (layer["kernel"]["x"], layer["kernel"]["y"]), activation=layer["activation"]))
			elif layer["name"] == "MaxPooling2D":
				model.add(MaxPooling2D((layer["kernel"]["x"], layer["kernel"]["y"]), padding=layer["padding"]))
			elif layer["name"] == "Flatten":
				model.add(Flatten())
		except:
			history.steps_info = {"message": "Error dimensions hidden layers"}
			history.save()
			prog.delete()
			return

	model.compile(model_info["optimizer"], model_info["loss"], ["accuracy"])

	prog.running = True
	prog.save()

	callback = LossHistory(prog)
	try:
		model.fit(x_train, y_train, batch_size=int(model_info["batchSize"]), epochs=int(model_info["epochs"]), callbacks=[callback], verbose=2)
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

	saved_model_path = save_keras_model(model, model_path, as_text=True)

	history.path = saved_model_path.decode()
	history.accuracy = accuracy[1]
	history.loss = accuracy[0]
	history.steps_info = callback.losses
	history.save()

	prog.delete()


@shared_task
def build_model_unsupervised(model_info: dict, user, model_path: str):
	print(model_info)

	x_train, x_test, y_train, y_test, input_shape = get_training_data(model_info["dataset"])

	if model_info["mnist"]:
		x_train = x_train.reshape(-1, 28 * 28)
		x_test = x_test.reshape(-1, 28 * 28)

	prog, history = init_prog_hist(user, model_info["dataset"])

	n_features = np.shape(x_train)[1]
	model_info["input4encoder"] = n_features

	encoder = train_autoencoder(x_train, model_info)
	model = build_clustering_model(x_train, encoder, model_info["n_clusters"])

	for i in range(model_info["epochs"]):  # walk through iterations
		predictions = model.predict(x_train)  # predict clusters
		weights = predictions ** 2 / predictions.sum(0)  # calculate
		targets = (weights.T / weights.sum(1)).T  # target distribution
		model.train_on_batch(x=x_train, y=targets)  # update model weights

	saved_model_path = save_keras_model(model, model_path, as_text=True)

	accuracy = model.evaluate(x_test, y_test)

	history.path = saved_model_path.decode()
	history.accuracy = accuracy[1]
	history.loss = accuracy[0]
	history.steps_info = []
	history.save()

	prog.delete()

