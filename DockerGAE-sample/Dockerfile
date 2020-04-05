FROM tensorflow/tensorflow:1.14.0-py3

ADD static/ ./static
ADD templates/ ./templates
ADD datasets/ ./datasets
ADD user_inputs/ ./user_inputs
ADD requirements.txt ./
ADD main.py ./
ADD loadmodel.py ./
ADD app.yaml ./
ADD README.md ./

RUN pip install -r requirements.txt

RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]

EXPOSE 8080

CMD ["python", "main.py"]