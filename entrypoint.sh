#!/bin/bash

################################
# check environment variables
################################
required_env_vars=(
	DEFAULT_ARTIFACT_ROOT
	AWS_ACCESS_KEY_ID
	AWS_SECRET_ACCESS_KEY
	POSTGRES_HOST
	POSTGRES_PORT
	POSTGRES_USERNAME
	POSTGRES_PASSWORD
	POSTGRES_DATABASE
)

missing_vars=()

for var in "${required_env_vars[@]}"; do
	if [ -z "${!var}" ]; then
		missing_vars+=("$var")
	fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
	echo "Error: The following environment variables are not set:"
	for missing_var in "${missing_vars[@]}"; do
		echo "  $missing_var"
	done
	exit 1
else
	echo "All required environment variables are set."
fi

################################
# check s3 permissions
################################
echo "Test file content" >test_file.txt

if [ -n "$MLFLOW_S3_ENDPOINT_URL" ]; then
	aws s3 --endpoint-url "$MLFLOW_S3_ENDPOINT_URL" ls $DEFAULT_ARTIFACT_ROOT >/dev/null 2>&1
	READ_PERMISSION=$?

	aws s3 --endpoint-url "$MLFLOW_S3_ENDPOINT_URL" cp test_file.txt "$DEFAULT_ARTIFACT_ROOT/test_file.txt" >/dev/null 2>&1
	WRITE_PERMISSION=$?
else
	aws s3 ls "$DEFAULT_ARTIFACT_ROOT" >/dev/null 2>&1
	READ_PERMISSION=$?

	aws s3 cp test_file.txt "$DEFAULT_ARTIFACT_ROOT/test_file.txt" >/dev/null 2>&1
	WRITE_PERMISSION=$?
fi

rm test_file.txt

if [ $READ_PERMISSION -eq 0 ] && [ $WRITE_PERMISSION -eq 0 ]; then
	echo "Read and write permissions are granted for the S3 bucket '$DEFAULT_ARTIFACT_ROOT'."

	################################
	# entrypoint
	################################
	postgres_uri="postgresql://$POSTGRES_USERNAME:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DATABASE"

	mlflow db upgrade "$postgres_uri"
	mlflow server \
		--backend-store-uri "$postgres_uri" \
		--artifacts-destination "$DEFAULT_ARTIFACT_ROOT" \
		--host 0.0.0.0 \
		--port 8080
else
	echo "Error: Insufficient permissions for the S3 bucket '$DEFAULT_ARTIFACT_ROOT'."
	exit 1
fi
