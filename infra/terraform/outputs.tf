output "ecr_repository_url" {
  description = "Where to push the image built from infra/Dockerfile.mlflow"
  value       = aws_ecr_repository.mlflow.repository_url
}

output "artifact_bucket" {
  description = "S3 bucket holding run artifacts (MLFLOW_ARTIFACT_BUCKET)"
  value       = aws_s3_bucket.artifacts.bucket
}

output "rds_address" {
  description = "Postgres host name (POSTGRES_HOST inside the task); not reachable from outside the VPC"
  value       = aws_db_instance.mlflow.address
}

output "ecs_cluster" {
  value = aws_ecs_cluster.mlflow.name
}

output "ecs_service" {
  value = aws_ecs_service.mlflow.name
}

output "log_group" {
  description = "CloudWatch log group, 14-day retention"
  value       = aws_cloudwatch_log_group.mlflow.name
}

output "push_image" {
  description = "Build the image on the Mac and push it; run from the repo root after `terraform apply`"
  value       = <<-EOT
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.mlflow.repository_url}
    docker build --platform linux/${lower(var.cpu_architecture) == "arm64" ? "arm64" : "amd64"} --build-arg MLFLOW_VERSION=${var.mlflow_version} -t ${aws_ecr_repository.mlflow.repository_url}:${var.mlflow_version} -f infra/Dockerfile.mlflow infra
    docker push ${aws_ecr_repository.mlflow.repository_url}:${var.mlflow_version}
    aws ecs update-service --region ${var.aws_region} --cluster ${aws_ecs_cluster.mlflow.name} --service ${aws_ecs_service.mlflow.name} --force-new-deployment
  EOT
}

output "server_ip_command" {
  description = "The task's public IP changes on every restart; this prints the current one. MLflow is then at http://<ip>:<mlflow_port>"
  value       = <<-EOT
    TASK=$(aws ecs list-tasks --region ${var.aws_region} --cluster ${aws_ecs_cluster.mlflow.name} --service-name ${aws_ecs_service.mlflow.name} --query 'taskArns[0]' --output text)
    ENI=$(aws ecs describe-tasks --region ${var.aws_region} --cluster ${aws_ecs_cluster.mlflow.name} --tasks $TASK --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
    aws ec2 describe-network-interfaces --region ${var.aws_region} --network-interface-ids $ENI --query 'NetworkInterfaces[0].Association.PublicIp' --output text
  EOT
}
