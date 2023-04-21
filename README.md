# python-vector-ai
Messing with langchain, huggingface transformers, and Elasticsearch

Set up your environment

```sh
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

or on windows
```sh
python.exe -m venv env
.\env\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements-win.txt
```


make a run.sh script that embeds your huggingface read api token
Don't check this into git

```sh
#!/bin/sh

export HUGGINGFACEHUB_API_TOKENs="YOUR TOKEN"

export ES_SERVER="YOURDESSERVERNAME.es.us-central1.gcp.cloud.es.io"
export ES_USERNAME="YOUR READ WRITE AND INDEX CREATING USER"
export ES_PASSWORD="YOUR PASSWORD"

python3 app-lotr.py
```

or on windows make a run.bat
```bat
@echo off
set ES_SERVER=XXXX
set ES_USERNAME=XXXX
set ES_PASSWORD=XXXX
python.exe app-lotr.py

```



run that script