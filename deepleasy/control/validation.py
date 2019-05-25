from .options import *


def model_builder_ok(data: dict):
	mandatory_params = {"epochs", "batchSize", "loss", "optimizer", "dataset", "trainPercentage"}

	if not mandatory_params.issubset(set(data.keys())):
		return False

	if data["loss"] not in LOSSES or data["optimizer"] not in OPTIMIZERS or data["dataset"] not in DATASETS:
		return False
	if data["batchSize"] < 0 or data["trainPercentage"] < 0.0 or data["trainPercentage"] > 1.0 or data["epochs"] < 1:
		return False

	if data.get("layers") is None or not isinstance(data["layers"], list):
		return False

	try:
		for layer in data["layers"]:
			if not layer_checker(layer):
				return False
		return True
	except:
		return False


def layer_checker(layer: dict):
	try:
		if layer["name"] == "Dense":
			if activation_checker(layer["activation"]) and units_checker(layer["units"]):
				return True

		elif layer["name"] == "Dropout":
			if rate_checker(layer["rate"]):
				return True

		elif layer["name"] == "Conv2D":
			if kernel_checker(layer["kernel"]) and activation_checker(layer["activation"]) and units_checker(
					layer["units"]):
				return True

		elif layer["name"] == "MaxPooling2D":
			if kernel_checker(layer["kernel"]) and padding_checker(layer["padding"]):
				return True

		elif layer["name"] == "Flatten":
			return True
		else:
			return False
	except:
		return False


def units_checker(units):
	return units > 0


def activation_checker(activation):
	return activation in ACTIVATIONS


def rate_checker(rate):
	return 0.0 < rate < 1.0


def kernel_checker(kernel):
	return kernel["x"] > 0 and kernel["y"] > 0


def padding_checker(padding):
	return padding == "valid" or padding == "same"
