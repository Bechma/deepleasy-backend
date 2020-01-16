FROM python:3.7
#FROM tensorflow/tensorflow:latest-py3
EXPOSE 8000

WORKDIR /usr/src/app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c 'import random; open("secret.txt", "w").write("".join([random.choice("abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)") for i in range(50)]))'
RUN python deepleasy/datasets.py

RUN apt update
RUN apt install redis-server -y
RUN service redis-server start

CMD [ "python", "./manage.py", "runserver", "0.0.0.0:8000" ]
