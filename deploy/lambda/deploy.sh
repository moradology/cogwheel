#!/usr/bin/env bash
set -euo pipefail

# One-command AWS Lambda deployer for cogzilla (provided.al2023, arm64 Graviton2).
# Requires only: aws CLI (configured) + curl. No Docker, SAM, or Rust required.
#
# Usage:
#   ./deploy.sh                                    # deploy public demo (latest release)
#   ./deploy.sh --bucket my-cogs                   # grant s3:GetObject on s3://my-cogs/*
#   ./deploy.sh --allow-prefixes s3://my-cogs/     # lock down allowed URL prefixes
#   ./deploy.sh --version v0.5.41 --name my-tiler   # custom version / function name
#   ./deploy.sh --zip /path/to/bootstrap.zip       # deploy a locally built zip
#   ./deploy.sh --destroy                          # delete the Lambda function

NAME="cogzilla"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
VERSION="latest"
MEMORY=1769
CACHE_MB=256
DISK_CACHE_MB=256
BUCKET=""
ALLOW_PREFIXES=""
ZIP_PATH=""
DESTROY=false
REPO="moradology/cogzilla"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)           NAME="$2"; shift 2 ;;
    --region)         REGION="$2"; shift 2 ;;
    --version)        VERSION="$2"; shift 2 ;;
    --memory)         MEMORY="$2"; shift 2 ;;
    --cache-mb)       CACHE_MB="$2"; shift 2 ;;
    --disk-cache-mb)  DISK_CACHE_MB="$2"; shift 2 ;;
    --bucket)         BUCKET="$2"; shift 2 ;;
    --allow-prefixes) ALLOW_PREFIXES="$2"; shift 2 ;;
    --zip)            ZIP_PATH="$2"; shift 2 ;;
    --destroy)        DESTROY=true; shift ;;
    -h|--help)
      sed -n '3,14p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

command -v aws >/dev/null || { echo "Error: aws CLI is required (https://aws.amazon.com/cli/)" >&2; exit 1; }

if [[ "$DESTROY" == true ]]; then
  echo "Deleting Lambda function '$NAME' in $REGION..."
  aws lambda delete-function-url-config --region "$REGION" --function-name "$NAME" 2>/dev/null || true
  aws lambda delete-function --region "$REGION" --function-name "$NAME"
  echo "Deleted."
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# 1. Obtain bootstrap zip
if [[ -z "$ZIP_PATH" ]]; then
  if [[ "$VERSION" == "latest" ]]; then
    VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
      | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)"
    [[ -n "$VERSION" ]] || { echo "Error: could not resolve latest release tag" >&2; exit 1; }
  fi
  URL="https://github.com/$REPO/releases/download/$VERSION/cogzilla-lambda-$VERSION-aarch64-unknown-linux-gnu.zip"
  ZIP_PATH="$TMP_DIR/cogzilla-lambda.zip"
  echo "Downloading $URL..."
  curl -fL --progress-bar "$URL" -o "$ZIP_PATH"
fi

# 2. Ensure minimal IAM execution role exists
ROLE_NAME="${NAME}-lambda-role"
if ! ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null)"; then
  echo "Creating IAM role '$ROLE_NAME'..."
  TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
  ROLE_ARN="$(aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST" \
    --query 'Role.Arn' --output text)"
  aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "Waiting 8s for IAM role propagation..."
  sleep 8
fi

if [[ -n "$BUCKET" ]]; then
  echo "Granting s3:GetObject on s3://$BUCKET/*..."
  POLICY="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\"],\"Resource\":\"arn:aws:s3:::${BUCKET}/*\"}]}"
  aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "${NAME}-s3-read" \
    --policy-document "$POLICY"
fi

# 3. Build environment variables JSON
ENV_VARS="COGZILLA_CACHE_MB=${CACHE_MB},COGZILLA_DISK_CACHE_MB=${DISK_CACHE_MB},COGZILLA_DISK_CACHE_DIR=/tmp/cogzilla-cache,COGZILLA_CACHE_CONTROL_MAX_AGE=3600,RUST_LOG=info"
if [[ -n "$ALLOW_PREFIXES" ]]; then
  ENV_VARS="${ENV_VARS},COGZILLA_ALLOWED_URL_PREFIXES=${ALLOW_PREFIXES}"
fi

# 4. Create or update Lambda function
if aws lambda get-function --region "$REGION" --function-name "$NAME" >/dev/null 2>&1; then
  echo "Updating existing Lambda function '$NAME' in $REGION..."
  aws lambda update-function-code \
    --region "$REGION" \
    --function-name "$NAME" \
    --zip-file "fileb://$ZIP_PATH" \
    --architectures arm64 >/dev/null
  aws lambda wait function-updated --region "$REGION" --function-name "$NAME"
  aws lambda update-function-configuration \
    --region "$REGION" \
    --function-name "$NAME" \
    --memory-size "$MEMORY" \
    --timeout 30 \
    --environment "Variables={${ENV_VARS}}" >/dev/null
else
  echo "Creating Lambda function '$NAME' (provided.al2023, arm64, ${MEMORY} MiB) in $REGION..."
  aws lambda create-function \
    --region "$REGION" \
    --function-name "$NAME" \
    --runtime provided.al2023 \
    --architectures arm64 \
    --handler bootstrap \
    --role "$ROLE_ARN" \
    --zip-file "fileb://$ZIP_PATH" \
    --memory-size "$MEMORY" \
    --timeout 30 \
    --environment "Variables={${ENV_VARS}}" >/dev/null
fi
aws lambda wait function-active-v2 --region "$REGION" --function-name "$NAME"

# 5. Ensure public HTTPS Function URL with CORS exists
if ! FUNC_URL="$(aws lambda get-function-url-config --region "$REGION" --function-name "$NAME" --query 'FunctionUrl' --output text 2>/dev/null)"; then
  echo "Creating public HTTPS Function URL..."
  FUNC_URL="$(aws lambda create-function-url-config \
    --region "$REGION" \
    --function-name "$NAME" \
    --auth-type NONE \
    --cors 'AllowOrigins=["*"],AllowMethods=["GET","POST"]' \
    --query 'FunctionUrl' --output text)"
  aws lambda add-permission \
    --region "$REGION" \
    --function-name "$NAME" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE >/dev/null 2>&1 || true
fi

FUNC_URL="${FUNC_URL%/}"
echo ""
echo "cogzilla is live on AWS Lambda!"
echo "  Demo page:  ${FUNC_URL}/"
echo "  Health:     ${FUNC_URL}/healthz"
echo "  Tile URL:   ${FUNC_URL}/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=s3://..."
echo ""
echo "Quick smoke test:"
echo "  curl -i ${FUNC_URL}/healthz"
