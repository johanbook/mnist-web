FROM python:latest
RUN pip install pipenv
COPY . .
RUN pipenv install --deploy --system
