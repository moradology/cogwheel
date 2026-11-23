# Deploying `cogzilla` on AWS Lambda

`cogzilla-lambda` runs natively on AWS Lambda (`provided.al2023`, `arm64` Graviton2) as a ~10 MiB `.zip` with ~15–25 ms cold starts, automatic binary tile encoding (`PNG`/`WebP`/`JPEG`/`f32`), and warm-container reuse of the decoded block cache, S3 TLS pool, and `/tmp` disk cache.

## Option 1: One-command deploy (no Rust, Docker, or SAM needed)

Requires only a configured `aws` CLI:

```bash
# Public demo (downloads the latest arm64 release zip, creates IAM role + Function URL)
./deploy/lambda/deploy.sh --region us-west-2

# Private S3 bucket + URL allowlist (recommended for production)
./deploy/lambda/deploy.sh \
  --region us-west-2 \
  --bucket my-cogs-bucket \
  --allow-prefixes "s3://my-cogs-bucket/,https://my-cogs-bucket.s3.amazonaws.com/"
```

When it finishes, open the printed Function URL in your browser to see the interactive demo page immediately.

To tear it down:

```bash
./deploy/lambda/deploy.sh --region us-west-2 --destroy
```

## Option 2: AWS SAM

Download the prebuilt arm64 zip from the [releases page](https://github.com/moradology/cogzilla/releases/latest) and deploy with [template.yaml](template.yaml):

```bash
mkdir -p dist
curl -fL https://github.com/moradology/cogzilla/releases/latest/download/cogzilla-lambda-v0.3.1-aarch64-unknown-linux-gnu.zip \
  -o dist/cogzilla-lambda-aarch64.zip

sam deploy \
  --template-file deploy/lambda/template.yaml \
  --stack-name cogzilla \
  --capabilities CAPABILITY_IAM \
  --resolve-s3
```

## Option 3: Terraform

```hcl
resource "aws_lambda_function" "cogzilla" {
  function_name = "cogzilla"
  role          = aws_iam_role.cogzilla.arn
  runtime       = "provided.al2023"
  architectures = ["arm64"]
  handler       = "bootstrap"
  filename      = "cogzilla-lambda-v0.3.1-aarch64-unknown-linux-gnu.zip"
  memory_size   = 1769 # 1 full Graviton2 vCPU
  timeout       = 30

  environment {
    variables = {
      COGZILLA_CACHE_MB              = "256"
      COGZILLA_DISK_CACHE_MB         = "256"
      COGZILLA_DISK_CACHE_DIR        = "/tmp/cogzilla-cache"
      COGZILLA_CACHE_CONTROL_MAX_AGE = "3600"
      COGZILLA_ALLOWED_URL_PREFIXES  = "s3://my-cogs-bucket/"
    }
  }
}

resource "aws_lambda_function_url" "cogzilla" {
  function_name      = aws_lambda_function.cogzilla.function_name
  authorization_type = "NONE"
  cors {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST"]
  }
}
```

## Configuration

| Env Variable | Default | Purpose |
| :--- | :--- | :--- |
| `COGZILLA_CACHE_MB` | `256` | In-memory decoded block cache (tier 1, RAM). Persists across warm invocations. |
| `COGZILLA_DISK_CACHE_MB` | `0` | Optional compressed-block cache in `/tmp` (tier 2). Set to `256` for warm-start reuse. |
| `COGZILLA_TILE_CACHE_MB` | `0` | Optional encoded-tile LRU micro-cache in RAM (`0` = singleflight dedup only). |
| `COGZILLA_ALLOWED_URL_PREFIXES` | *(open)* | Comma-separated URL prefixes (`s3://bucket/,https://...`). **Set this on public endpoints.** |
| `COGZILLA_CACHE_CONTROL_MAX_AGE` | `3600` | `Cache-Control: public, max-age=<N>` header for CloudFront/CDN caching. |
| `COGZILLA_MOSAIC` | | Optional `name=/path/to/catalog.json` pre-loaded at cold start. |

## Production Tips

- **Put CloudFront in front of the Function URL:** `cogzilla` emits strong `ETag`, `Cache-Control`, and honors `If-None-Match` (`304 Not Modified`), so CloudFront absorbs repeat map pans for pennies.
- **Deploy in the same AWS region as your S3 bucket:** keeps uncached S3 range GETs at ~2–5 ms instead of cross-region latency, and avoids cross-region data transfer fees.
- **Memory sizing:** Lambda allocates CPU linearly with RAM (`1,769 MiB` = 1 full vCPU, `3,008 MiB` = ~2 vCPUs). `1,769 MiB` with `COGZILLA_CACHE_MB=256` is the sweet spot for cost and tile latency.
- **Building locally for Lambda:** `cargo lambda build --release --arm64 --output-format zip -p cogzilla-lambda` (via [`cargo-lambda`](https://www.cargo-lambda.info/)).
