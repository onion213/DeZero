FROM python:3.9-slim-buster
RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip && pip install numpy