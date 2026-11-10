"""One-shot: make sure the artifact bucket exists on the local S3-compatible store (the `s3-init` compose service).
Runs from the same image as the server, so there is no extra image to pull. On AWS the bucket is made by Terraform and this never runs.
why: `mlflow server --artifacts-destination s3://bucket/` does not create the bucket; the first artifact upload would fail with NoSuchBucket."""
import os, sys, time
import boto3, botocore
bucket = os.environ["MLFLOW_ARTIFACT_BUCKET"]; endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
s3 = boto3.client("s3", endpoint_url=endpoint, config=botocore.config.Config(s3={"addressing_style": "path"}, retries={"max_attempts": 2}))
for attempt in range(30):   # why: the store's health check says "up" slightly before it accepts signed requests; a short retry beats a hard failure
    try:
        s3.head_bucket(Bucket=bucket); print(f"bucket {bucket} present at {endpoint}"); sys.exit(0)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            s3.create_bucket(Bucket=bucket); print(f"created bucket {bucket} at {endpoint}"); sys.exit(0)
        last = e
    except botocore.exceptions.EndpointConnectionError as e:
        last = e
    time.sleep(2)
print(f"giving up: {last}"); sys.exit(1)
