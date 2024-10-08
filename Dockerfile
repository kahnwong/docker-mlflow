FROM python:3.12-slim

# install aws cli
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y curl unzip --no-install-recommends \
    && apt-get clean \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf ./aws \
    && rm awscliv2.zip

# install mlflow
# hadolint ignore=DL3045
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app
WORKDIR /opt/mlflow
COPY entrypoint.sh .

EXPOSE 8080
CMD bash entrypoint.sh
