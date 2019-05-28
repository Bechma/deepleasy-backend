import os
import shutil
import tempfile

from django.http import HttpResponse
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from deepleasy.control.options import *
from deepleasy.control.validation import model_builder_ok, clustering_checker
from deepleasy.models import Progress, History
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

		Progress(user=request.user, epochs=0, max_epochs=request.data["epochs"], running=False).save()

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

		Progress(user=request.user, epochs=0, max_epochs=request.data["epochs"], running=False).save()

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
		print("Hello")
		print(request.data)

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

		for h in history:
			registers["history"].append({
				"id": h.id,
				"status": os.path.isdir(h.path),
				"path": h.path,
				"accuracy": h.accuracy,
				"loss": h.loss,
				"dataset": h.dataset,
				"steps_info": h.steps_info
			})

		return Response(registers)

	def post(self, request: Request):
		try:
			history = History.objects.get(id=request.data["id"])
			if os.path.exists(history.path):
				shutil.rmtree(history.path)
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
			"datasets": DATASETS,
			"optimizers": OPTIMIZERS
		})


class ModelGetter(APIView):
	permission_classes = (IsAuthenticated, )

	def post(self, request: Request):
		try:
			tmp = tempfile.TemporaryFile()
			h = History.objects.get(id=request.data["id"])
			shutil.make_archive(tmp.name, 'zip', h.path)
			i = open(tmp.name+'.zip', 'rb')
			r = HttpResponse(i, content_type='application/zip')
			r['Content-Disposition'] = 'attachment; filename="Model.zip"'
			return r
		except:
			return Response("Error", 404)


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
