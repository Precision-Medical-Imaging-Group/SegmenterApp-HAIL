FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git

COPY ./hail /tmp/hail
COPY ./model_weights /tmp/model_weights
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /tmp/hail

ENTRYPOINT ["python", "harmonizer.py"]