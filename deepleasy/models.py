import json

from django.contrib.auth.models import User
from django.db import models


# Create your models here.
class Progress(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE, to_field='username')
	epochs = models.IntegerField()
	max_epochs = models.IntegerField()
	running = models.BooleanField()
	task_id = models.TextField(default="")


class JsonField(models.TextField):
	"""
	Stores json-able python objects as json.
	"""

	def get_db_prep_value(self, value, connection, prepared=False):
		return json.dumps(value)

	def from_db_value(self, value, expression, connection, context):
		if value == "":
			return None
		return json.loads(value)


class History(models.Model):
	user = models.ForeignKey(User, on_delete=models.CASCADE, to_field='username')
	path = models.TextField()
	accuracy = models.FloatField(default=0.0)
	loss = models.FloatField(default=0.0)
	steps_info = JsonField()
	dataset = models.TextField()
