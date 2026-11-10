# Compose service -> AWS resource:  postgres -> RDS Postgres 16 · s3 -> S3 bucket · mlflow -> one ECS Fargate task from ECR
# · s3-init -> nothing (Terraform makes the bucket) · docker volumes -> RDS storage + the bucket · docker logs -> CloudWatch (14 days)
# Networking stays in the account's default VPC: the task gets a public IP and a security group that admits var.allowed_cidr only.
# No load balancer, no TLS, no NAT, no domain: each would cost more per month than the server itself (see docs/mlops/infra.md).

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---- artifact store: the s3 service ------------------------------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = var.mlflow_artifact_bucket
  # why: force_destroy lets `terraform destroy` remove a bucket that holds artifacts; the bucket is versioned, so a mistaken
  # delete of one object is recoverable, and a destroy of the whole stack is a deliberate act anyway
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  # why: versioning keeps every overwritten checkpoint forever unless old versions expire; 30 days is long enough to undo a mistake
  rule {
    id     = "tidy"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# ---- image registry ----------------------------------------------------------------------------------------------------------------
resource "aws_ecr_repository" "mlflow" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

# ---- logs --------------------------------------------------------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${var.project}"
  retention_in_days = 14
}

# ---- network: who may talk to whom -------------------------------------------------------------------------------------------------
resource "aws_security_group" "mlflow" {
  name        = "${var.project}-server"
  description = "MLflow server: admits var.allowed_cidr on the MLflow port only"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    description = "MLflow UI and API from the lab"
    from_port   = var.mlflow_port
    to_port     = var.mlflow_port
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  egress {
    description = "ECR pulls, S3, RDS, SSM"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "db" {
  name        = "${var.project}-db"
  description = "RDS: admits the MLflow task only"
  vpc_id      = data.aws_vpc.default.id
  ingress {
    description     = "Postgres from the MLflow task"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.mlflow.id]
  }
}

# ---- backend store: the postgres service ------------------------------------------------------------------------------------------
resource "aws_db_subnet_group" "db" {
  name       = var.project
  subnet_ids = data.aws_subnets.default.ids
}

# THE recurring cost. db.t4g.micro, single-AZ, PostgreSQL, us-west-2 on-demand: USD 0.016/hour = USD 11.68 per 730-hour month,
# plus gp3 storage at about USD 0.115/GB-month (20 GB = USD 2.30). Looked up 10 Nov 2026 on sparecores.com and holori.com
# (AWS's own pricing page hides the table behind scripts; confirm in the AWS Pricing Calculator before relying on it).
resource "aws_db_instance" "mlflow" {
  identifier              = var.project
  engine                  = "postgres"
  engine_version          = "16"
  instance_class          = var.db_instance_class
  allocated_storage       = 20
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = var.postgres_db
  username                = var.postgres_user
  password                = var.postgres_password
  multi_az                = false
  publicly_accessible     = false
  db_subnet_group_name    = aws_db_subnet_group.db.name
  vpc_security_group_ids  = [aws_security_group.db.id]
  backup_retention_period = 7
  # why: a one-person lab that may tear the whole thing down; deletion protection and a final snapshot would only get in the way
  deletion_protection        = false
  skip_final_snapshot        = true
  apply_immediately          = true
  auto_minor_version_upgrade = true
}

# why: the task reads the password from SSM at start (ECS `secrets`), so the task definition JSON never contains it
resource "aws_ssm_parameter" "postgres_password" {
  name  = "/${var.project}/POSTGRES_PASSWORD"
  type  = "SecureString"
  value = var.postgres_password
}

# ---- IAM: pull the image and write logs (execution role); read and write the bucket (task role) -------------------------------------
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.project}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "execution_secret" {
  statement {
    actions   = ["ssm:GetParameters"]
    resources = [aws_ssm_parameter.postgres_password.arn]
  }
}

resource "aws_iam_role_policy" "execution_secret" {
  name   = "read-postgres-password"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secret.json
}

resource "aws_iam_role" "task" {
  name               = "${var.project}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task_bucket" {
  statement {
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
}

resource "aws_iam_role_policy" "task_bucket" {
  name   = "artifact-bucket"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_bucket.json
}

# ---- the server: the mlflow service -------------------------------------------------------------------------------------------------
resource "aws_ecs_cluster" "mlflow" {
  name = var.project
}

resource "aws_ecs_task_definition" "mlflow" {
  family                   = var.project
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 256 # 0.25 vCPU: the smallest Fargate size
  memory                   = 512 # 0.5 GB
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = var.cpu_architecture
  }
  # Same image, same entrypoint.sh, same variable names as docker-compose.yml. No MLFLOW_S3_ENDPOINT_URL and no access keys:
  # boto3 finds real S3 by region and the task role's credentials by itself.
  container_definitions = jsonencode([{
    name      = "mlflow"
    image     = "${aws_ecr_repository.mlflow.repository_url}:${var.mlflow_version}"
    essential = true
    portMappings = [{
      containerPort = var.mlflow_port
      protocol      = "tcp"
    }]
    environment = [
      { name = "MLFLOW_PORT", value = tostring(var.mlflow_port) },
      { name = "MLFLOW_WORKERS", value = "1" }, # why: each worker holds ~150 MB; one is right for 0.5 GB and a one-person lab
      { name = "POSTGRES_HOST", value = aws_db_instance.mlflow.address },
      { name = "POSTGRES_PORT", value = tostring(aws_db_instance.mlflow.port) },
      { name = "POSTGRES_DB", value = var.postgres_db },
      { name = "POSTGRES_USER", value = var.postgres_user },
      { name = "MLFLOW_ARTIFACT_BUCKET", value = aws_s3_bucket.artifacts.bucket },
      { name = "AWS_DEFAULT_REGION", value = var.aws_region },
      { name = "MLFLOW_ALLOWED_HOSTS", value = "*" }, # why: the task is reached by a public IP MLflow 3 would otherwise reject; the security group is the lock
    ]
    secrets = [
      { name = "POSTGRES_PASSWORD", valueFrom = aws_ssm_parameter.postgres_password.arn },
    ]
    healthCheck = {
      command     = ["CMD", "python", "/app/healthcheck.py"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.mlflow.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "mlflow"
      }
    }
  }])
}

resource "aws_ecs_service" "mlflow" {
  name            = var.project
  cluster         = aws_ecs_cluster.mlflow.id
  task_definition = aws_ecs_task_definition.mlflow.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  # why: one writer at a time against one small database; a deploy stops the old task before starting the new one
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100
  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.mlflow.id]
    assign_public_ip = true # why: no NAT gateway (USD 32+/month); the public IP is how the task reaches ECR, S3 and SSM, and how the lab reaches it
  }
  depends_on = [aws_iam_role_policy_attachment.execution, aws_iam_role_policy.execution_secret]
}
