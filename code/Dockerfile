FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git

COPY . /tmp/hail

RUN pip install /tmp/hail && rm -rf /tmp/hail

ENTRYPOINT ["hail"]