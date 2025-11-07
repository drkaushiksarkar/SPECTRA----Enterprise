FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY . /work
RUN pip install -U pip && pip install -e . && pip cache purge
ENTRYPOINT ["bash"]
