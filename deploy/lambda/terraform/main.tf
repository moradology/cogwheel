terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = ">= 3.2"
    }
  }
}

locals {
  arch_triple = var.architecture == "arm64" ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu"
  release_url = "https://github.com/moradology/cogzilla/releases/download/${var.cogzilla_version}/cogzilla-lambda-${var.cogzilla_version}-${local.arch_triple}.zip"
  zip_file    = var.zip_path != "" ? var.zip_path : "${path.module}/.cogzilla-lambda-${var.cogzilla_version}-${var.architecture}.zip"

  base_env = {
    COGZILLA_CACHE_MB              = tostring(var.cache_mb)
    COGZILLA_DISK_CACHE_MB         = tostring(var.disk_cache_mb)
    COGZILLA_DISK_CACHE_DIR        = "/tmp/cogzilla-cache"
    COGZILLA_TILE_CACHE_MB         = tostring(var.tile_cache_mb)
    COGZILLA_CACHE_CONTROL_MAX_AGE = tostring(var.cache_control_max_age)
    RUST_LOG                       = var.log_level
  }
  allowlist_env = length(var.allowed_url_prefixes) > 0 ? {
    COGZILLA_ALLOWED_URL_PREFIXES = join(",", var.allowed_url_prefixes)
  } : {}
  env_vars = merge(local.base_env, local.allowlist_env, var.extra_env)
}

# Automatically fetch the prebuilt release zip when zip_path is not provided.
resource "null_resource" "download_release" {
  count = var.zip_path == "" ? 1 : 0

  triggers = {
    version      = var.cogzilla_version
    architecture = var.architecture
  }

  provisioner "local-exec" {
    command = "curl -fsSL '${local.release_url}' -o '${local.zip_file}'"
  }
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "cogzilla" {
  name               = "${var.function_name}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "basic_logs" {
  role       = aws_iam_role.cogzilla.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "s3_read" {
  count = length(var.s3_bucket_arns) > 0 ? 1 : 0
  statement {
    actions   = ["s3:GetObject"]
    resources = [for arn in var.s3_bucket_arns : "${trimsuffix(arn, "/")}/*"]
  }
}

resource "aws_iam_role_policy" "s3_read" {
  count  = length(var.s3_bucket_arns) > 0 ? 1 : 0
  name   = "${var.function_name}-s3-read"
  role   = aws_iam_role.cogzilla.id
  policy = data.aws_iam_policy_document.s3_read[0].json
}

resource "aws_lambda_function" "cogzilla" {
  function_name = var.function_name
  role          = aws_iam_role.cogzilla.arn
  runtime       = "provided.al2023"
  architectures = [var.architecture]
  handler       = "bootstrap"
  filename      = local.zip_file
  memory_size   = var.memory_size
  timeout       = var.timeout

  ephemeral_storage {
    size = var.ephemeral_storage_mb
  }

  environment {
    variables = local.env_vars
  }

  tags       = var.tags
  depends_on = [null_resource.download_release, aws_iam_role_policy_attachment.basic_logs]
}

resource "aws_lambda_function_url" "cogzilla" {
  function_name      = aws_lambda_function.cogzilla.function_name
  authorization_type = var.function_url_auth_type

  cors {
    allow_origins = var.cors_allow_origins
    allow_methods = ["GET", "POST"]
    allow_headers = ["*"]
  }
}

# Optional CloudFront CDN in front of the Function URL (honors cogzilla ETag + Cache-Control).
resource "aws_cloudfront_cache_policy" "cogzilla" {
  count       = var.enable_cloudfront ? 1 : 0
  name        = "${var.function_name}-tiles-cache-policy"
  default_ttl = var.cache_control_max_age
  max_ttl     = 86400
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    cookies_config {
      cookie_behavior = "none"
    }
    headers_config {
      header_behavior = "none"
    }
    query_strings_config {
      query_string_behavior = "all"
    }
    enable_accept_encoding_brotli = false
    enable_accept_encoding_gzip   = false
  }
}

resource "aws_cloudfront_distribution" "cogzilla" {
  count           = var.enable_cloudfront ? 1 : 0
  enabled         = true
  is_ipv6_enabled = true
  comment         = "${var.function_name} COG tile CDN"
  price_class     = var.cloudfront_price_class

  origin {
    origin_id   = "cogzilla-lambda"
    domain_name = trimsuffix(replace(aws_lambda_function_url.cogzilla.function_url, "https://", ""), "/")

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = "cogzilla-lambda"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD", "OPTIONS"]
    cache_policy_id        = aws_cloudfront_cache_policy.cogzilla[0].id
    compress               = false
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = var.tags
}
