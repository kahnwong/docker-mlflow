# docker-mlflow

## Usage
```bash
make start # Access via http://localhost:5000
```

## Development

```bash
# build
docker build -t mlflow .

# run
docker run -p 8080:8080 --env-file .env mlflow
```
