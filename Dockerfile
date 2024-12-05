FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim


# install mlflow
# hadolint ignore=DL3045
COPY pyproject.toml uv.lock ./
RUN uv export --no-hashes --no-dev --no-emit-project --output-file=requirements.txt && \
  pip install --no-cache-dir -r requirements.txt

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

# app
WORKDIR /opt/mlflow
COPY entrypoint.sh .

EXPOSE 8080
CMD bash entrypoint.sh
