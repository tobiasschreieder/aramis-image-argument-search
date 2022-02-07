# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /aramis_imarg

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --default-timeout=5000

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN python -m spacy download en_core_web_sm
RUN python -c "exec(\"import nltk\nnltk.download('vader_lexicon')\")"

RUN apt-get install tesseract-ocr -y

COPY . .

ENTRYPOINT ["python", "startup.py"]