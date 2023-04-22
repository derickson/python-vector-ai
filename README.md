# python-vector-ai
Messing with langchain, huggingface transformers, and Elasticsearch

This is a branch holding the state of the app from April 2023 when I did the blog. Switch to main to branch main to see updates since then.

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
export ES_PASSWORD="YOUR PASSWORD"

python3 app-lotr.py
```

run that script