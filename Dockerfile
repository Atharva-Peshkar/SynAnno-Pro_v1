# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app
ENV PORT 8080
ENV HOST 0.0.0.0

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install --upgrade pip


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-u", "-m" , "flask", "run", "--host=0.0.0.0", "-p", "8080"]
