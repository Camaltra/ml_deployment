FROM python:3.11.3-slim-buster

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=app/project/__init__.py
ENV FLASK_DEBUG=1
ENV PYTHONPATH=/usr/src/app/src:

RUN apt-get update && apt-get install -y netcat

RUN pip install --upgrade pip
COPY ./app.requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

COPY . /usr/src/app/

CMD python app/manage.py run -h 0.0.0.0 -p 5000
