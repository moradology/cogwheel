variable "function_name" {
  type        = string
  default     = "cogzilla"
  description = "AWS Lambda function name."
}

variable "cogzilla_version" {
  type        = string
  default     = "v0.7.10"
  description = "Release tag to download from GitHub Releases (ignored when zip_path is set)."
}

variable "architecture" {
  type        = string
  default     = "arm64"
  description = "Lambda CPU architecture: arm64 (Graviton2, recommended) or x86_64."
  validation {
    condition     = contains(["arm64", "x86_64"], var.architecture)
    error_message = "architecture must be 'arm64' or 'x86_64'."
  }
}

variable "zip_path" {
  type        = string
  default     = ""
  description = "Optional path to a pre-downloaded or locally built bootstrap .zip. If empty, downloads from GitHub Releases."
}

variable "memory_size" {
  type        = number
  default     = 1769
  description = "Lambda memory in MiB (1769 MiB = 1 full Graviton2 vCPU)."
}

variable "timeout" {
  type        = number
  default     = 30
  description = "Lambda timeout in seconds."
}

variable "ephemeral_storage_mb" {
  type        = number
  default     = 512
  description = "Lambda /tmp storage in MiB (used when disk_cache_mb > 0)."
}

variable "cache_mb" {
  type        = number
  default     = 256
  description = "In-memory decoded block cache size in MiB."
}

variable "disk_cache_mb" {
  type        = number
  default     = 256
  description = "Tier-2 compressed-block cache in /tmp (MiB). Set 0 to disable."
}

variable "tile_cache_mb" {
  type        = number
  default     = 0
  description = "Optional encoded-tile LRU micro-cache in RAM (MiB)."
}

variable "cache_control_max_age" {
  type        = number
  default     = 3600
  description = "Cache-Control max-age in seconds on tile responses."
}

variable "s3_bucket_arns" {
  type        = list(string)
  default     = []
  description = "S3 bucket ARNs (e.g. ['arn:aws:s3:::my-cogs']) to grant read-only s3:GetObject access."
}

variable "allowed_url_prefixes" {
  type        = list(string)
  default     = []
  description = "Optional COGZILLA_ALLOWED_URL_PREFIXES allowlist (recommended on public endpoints)."
}

variable "function_url_auth_type" {
  type        = string
  default     = "NONE"
  description = "Lambda Function URL auth type: NONE or AWS_IAM."
}

variable "cors_allow_origins" {
  type        = list(string)
  default     = ["*"]
  description = "CORS AllowOrigins for the Lambda Function URL."
}

variable "enable_cloudfront" {
  type        = bool
  default     = false
  description = "Provision a CloudFront CDN distribution in front of the Function URL."
}

variable "cloudfront_price_class" {
  type        = string
  default     = "PriceClass_100"
  description = "CloudFront price class when enable_cloudfront = true."
}

variable "log_level" {
  type        = string
  default     = "info"
  description = "RUST_LOG level (info, warn, debug)."
}

variable "extra_env" {
  type        = map(string)
  default     = {}
  description = "Additional environment variables (e.g. AWS_REQUEST_PAYER, GS_USER_PROJECT, AZURE_STORAGE_SAS_TOKEN)."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Tags applied to all created AWS resources."
}
