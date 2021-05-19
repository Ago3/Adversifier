FROM python:3.8

WORKDIR /aaa

copy . /aaa

RUN pip install --upgrade pip && pip install nltk numpy sklearn requests
