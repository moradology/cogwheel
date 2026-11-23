output "function_url" {
  description = "Direct HTTPS Lambda Function URL (open in browser for the interactive demo page)."
  value       = aws_lambda_function_url.cogzilla.function_url
}

output "cloudfront_url" {
  description = "CloudFront CDN HTTPS URL (when enable_cloudfront = true)."
  value       = var.enable_cloudfront ? "https://${aws_cloudfront_distribution.cogzilla[0].domain_name}" : null
}

output "function_name" {
  description = "Deployed Lambda function name."
  value       = aws_lambda_function.cogzilla.function_name
}

output "role_arn" {
  description = "IAM execution role ARN (attach additional S3/KMS policies here if needed)."
  value       = aws_iam_role.cogzilla.arn
}
