#!/usr/bin/env bash
# Build the API image and push it to Amazon ECR.
#
# Required env vars:
#   AWS_ACCOUNT_ID  - 12-digit AWS account id
#   AWS_REGION      - e.g. us-west-2
#   ECR_REPO        - ECR repository name, e.g. revenue-forecasting-api
# Optional:
#   IMAGE_TAG       - defaults to current git short SHA (or "latest")
#   PLATFORM        - defaults to linux/amd64 (ECS/Lambda/Fargate x86_64)
#
# Usage:
#   AWS_ACCOUNT_ID=123456789012 \
#   AWS_REGION=us-west-2 \
#   ECR_REPO=revenue-forecasting-api \

: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required}"
: "${AWS_REGION:?AWS_REGION is required}"
: "${ECR_REPO:?ECR_REPO is required}"

PLATFORM="${PLATFORM:-linux/amd64}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"
LATEST_URI="${REGISTRY}/${ECR_REPO}:latest"

# Repo root (one level up from this script)
cd "$(dirname "$0")/.."

echo ">> Ensuring ECR repository exists: ${ECR_REPO}"
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
       --image-scanning-configuration scanOnPush=true >/dev/null

echo ">> Logging in to ECR: ${REGISTRY}"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

echo ">> Building image: ${IMAGE_URI} (platform=${PLATFORM})"
# Use buildx so we can cross-build for linux/amd64 from an Apple Silicon Mac.
docker buildx build \
  --platform "${PLATFORM}" \
  --tag "${IMAGE_URI}" \
  --tag "${LATEST_URI}" \
  --provenance=false \
  --push \
  .

echo ">> Pushed:"
echo "   ${IMAGE_URI}"
echo "   ${LATEST_URI}"
