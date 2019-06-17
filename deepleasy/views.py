import io
import os
import zipfile

import h5py
import numpy as np
from PIL import Image
from django.http import HttpResponse
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from tensorflow.python.keras.engine.saving import load_model

from deepleasy.control.options import OPTIMIZERS, LOSSES, DATASETS, LAYERS, ACTIVATIONS
from deepleasy.control.validation import model_builder_ok, clustering_checker
from deepleasy.models import Progress, History
from deepleasy.training.clustering import ClusteringLayer
from deepleasy.training.train import build_model_supervised, build_model_unsupervised
from webtfg.celery import app

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


# Create your views here.
class ModelBuilderUnsupervised(APIView):
	permission_classes = (IsAuthenticated, )

	def post(self, request: Request):
		if not model_builder_ok(request.data) and clustering_checker(request.data):
			print(request.data)
			return Response({
				"success": False,
				"message": "The model had some problems whilst data verification, refresh the page and try again"
			})

		try:
			Progress(user=request.user, epochs=0, max_epochs=request.data["epochs"], running=False).save()
		except:
			return Response({
				"success": False,
				"message": "There is another model building, please wait until it's finished"
			})

		build_model_unsupervised.delay(request.data, request.user.username, model_path)
		print(request.user)

		return Response({
			"success": True,
			"message": "The model is built! Have a cup of tea while waiting :)"
		})


class ModelBuilderSupervised(APIView):
	permission_classes = (IsAuthenticated, )

	def post(self, request: Request):
		if not model_builder_ok(request.data):
			print(request.data)
			return Response({
				"success": False,
				"message": "The model had some problems whilst data verification, refresh the page and try again"
			})

		try:
			Progress(user=request.user, epochs=0, max_epochs=request.data["epochs"], running=False).save()
		except:
			return Response({
				"success": False,
				"message": "There is another model building, please wait until it's finished"
			})

		build_model_supervised.delay(request.data, request.user.username, model_path)
		print(request.user)

		return Response({
			"success": True,
			"message": "The model is built! Have a cup of tea while waiting :)"
		})


class ModelProgress(APIView):
	permission_classes = (IsAuthenticated,)

	def get(self, request: Request):
		try:
			progress = Progress.objects.get(user=request.user)
			return Response({
				"epochs": progress.epochs,
				"max_epochs": progress.max_epochs,
				"running": progress.running,
				"task_id": progress.task_id
			})
		except Progress.DoesNotExist:
			return Response({"message": "There is no active task right now"}, 404)

	def post(self, request: Request):
		try:
			app.control.revoke(task_id=request.data["task_id"], terminate=True, timeout=1, wait=False)
		except:
			return Response("problem removing", 502)

		Progress.objects.get(task_id=request.data["task_id"]).delete()
		History(user=request.user, path="error", accuracy=0.0, steps_info={"problem": "Cancelled by user"}).save()
		return Response("removed", 200)


class ModelHistory(APIView):
	permission_classes = (IsAuthenticated, )

	def get(self, request: Request):
		history = History.objects.filter(user=request.user)
		if len(history) == 0:
			return Response({"message": "There is no history registers for you :("}, 404)

		registers = {
			"history": []
		}

		for h in reversed(history):
			registers["history"].append({
				"id": h.id,
				"timestamp": h.timestamp,
				"status": os.path.exists(h.path),
				"accuracy": h.accuracy,
				"loss": h.loss,
				"dataset": h.dataset,
				"steps_info": h.steps_info
			})

		return Response(registers)

	def post(self, request: Request):
		try:
			history = History.objects.get(id=request.data["id"])
			if history.path != "" and os.path.exists(history.path):
				os.remove(history.path)
			history.delete()
			return Response("OK", 200)
		except:
			return Response("Error, data not found", 404)


class ModelOptions(APIView):
	permission_classes = (IsAuthenticated, )

	def get(self, request: Request):
		return Response({
			"layers": LAYERS,
			"losses": LOSSES,
			"datasets": [i.toJSON() for i in DATASETS],
			"optimizers": OPTIMIZERS,
			"activations": ACTIVATIONS
		})


class ModelGetter(APIView):
	permission_classes = (IsAuthenticated, )

	def post(self, request: Request):
		try:
			h = History.objects.get(id=request.data["id"])
			zip_buffer = io.BytesIO()
			with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
				# ModelGetter.zipdir(h.path, zip_file)
				arc, _ = os.path.split(h.path)
				zip_file.write(h.path, h.path[len(arc):])
			r = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
			r['Content-Disposition'] = 'attachment; filename="Model.zip"'
			return r
		except:
			return Response("Error", 404)

	@staticmethod
	def zipdir(path, ziph):
		# ziph is zipfile handle
		abs_src, _ = os.path.split(path)
		for root, dirs, files in os.walk(path):
			for file in files:
				absname = os.path.abspath(os.path.join(root, file))
				arcname = absname[len(abs_src) + 1:]
				ziph.write(absname, arcname=arcname)


class UserStats(APIView):
	permission_classes = (IsAuthenticated, )

	def get(self, request: Request):
		progress = None
		try:
			progress = Progress.objects.get(user=request.user)
		except:
			pass

		history = None
		try:
			history = History.objects.filter(user=request.user).count()
		except:
			pass

		response = {}
		if progress is not None:
			response["epochs"] = progress.epochs
			response["max_epochs"] = progress.max_epochs
			response["running"] = progress.running
		else:
			response["epochs"] = 0
			response["max_epochs"] = 0
			response["running"] = False

		if history is not None:
			response["history_entries"] = history
		else:
			response["history_entries"] = 0

		return Response(response)


class ModelPredict(APIView):
	permission_classes = (IsAuthenticated,)

	def post(self, request: Request):
		if "zippy" not in request.data.keys():
			return Response("Where is my zip?", 400)
		elif "dataset" not in request.data.keys():
			return Response("Where is the dataset?", 400)
		elif request.data["dataset"] not in [x.name for x in DATASETS]:
			return Response("Dataset not valid", 400)

		try:
			input_zip = zipfile.ZipFile(request.data["zippy"])
		except zipfile.BadZipFile:
			return Response("What kind of zip is this", 400)

		predictions = None
		model = None
		pnames = []

		for x in input_zip.namelist():
			if x.endswith(".h5"):
				file = h5py.File(io.BytesIO(input_zip.read(x)))
				model = load_model(file, custom_objects={'ClusteringLayer': ClusteringLayer})
			else:
				try:
					image = Image.open(io.BytesIO(input_zip.read(x)))
					image.load()
					image = np.asarray(image, dtype="float32") / 255
					try:
						image = image.reshape((28, 28, 1))
					except ValueError:
						return Response("image {} is not in good format".format(x), 400)
					if predictions is None:
						predictions = np.array([image])
					else:
						predictions = np.append(predictions, [image], axis=0)
					pnames.append(x)
				except IOError:
					pass
		if model is not None:
			predictions = model.predict(predictions)
			predictions = predictions.argmax(axis=1)
			print("PREDICTIONS", predictions)
			return Response({
				"predictions": [
					{"feature": n, "prediction": p} for n, p in zip(pnames, predictions)
				]
			})
		else:
			return Response("Where is the h5?", 400)
