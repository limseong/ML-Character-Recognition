# Sample Flask running on Docker + Google App Engine Flexible

## Dependancies (Using venv)
```
python -m venv env
env/Scripts/activate
pip install -r requirements.txt
```

## Run local
```
python main.py
```

## Build Docker
```
docker build -t <container-name> .
docker run -it -p 8080:8080 <container-name>
```

## Deploy to Google App Engine
With app.yaml
```
gcloud app deploy --project <project-id>
```
