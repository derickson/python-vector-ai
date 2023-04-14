# python-vector-ai
Messing with langchain, huggingface transformers, and Elasticsearch

Set up your environment

```sh
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

make a run.sh script that embeds your huggingface read api token
Don't check this into git

```sh
#!/bin/sh

export HUGGINGFACEHUB_API_TOKENs="YOUR TOKEN"

export ES_SERVER="YOURDESSERVERNAME.es.us-central1.gcp.cloud.es.io"
export ES_USERNAME="YOUR READ WRITE AND INDEX CREATING USER"
export ES_PASSOWRD="YOUR PASSWORD"

python3 app.py
```

run that script