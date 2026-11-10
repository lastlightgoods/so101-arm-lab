# Variable names mirror infra/.env.example in lower case (MLFLOW_PORT -> mlflow_port) so one set of settings describes both builds.
# Values without a default are the ones with no sensible shared answer: give them in terraform.tfvars (gitignored) or -var flags.

variable "aws_region" {
  description = "Region for everything. us-west-2 (Oregon) is the closest to Bend."
  type        = string
  default     = "us-west-2"
}

variable "project" {
  description = "Name prefix for every resource."
  type        = string
  default     = "so101-mlflow"
}

variable "mlflow_version" {
  description = "Image tag pushed to ECR; must equal the MLflow version in the Mac venv (MLFLOW_VERSION in infra/.env)."
  type        = string
  default     = "3.17.2"
}

variable "mlflow_port" {
  description = "Port the server listens on and the security group admits (MLFLOW_PORT)."
  type        = number
  default     = 5050
}

variable "postgres_db" {
  description = "Backend store database name (POSTGRES_DB)."
  type        = string
  default     = "mlflow"
}

variable "postgres_user" {
  description = "Backend store master user (POSTGRES_USER)."
  type        = string
  default     = "mlflow"
}

variable "postgres_password" {
  description = "Backend store master password (POSTGRES_PASSWORD). Stored in SSM as a SecureString for the task; never in a file in the repo."
  type        = string
  sensitive   = true
}

variable "mlflow_artifact_bucket" {
  description = "S3 bucket for artifacts (MLFLOW_ARTIFACT_BUCKET). Bucket names are global, so pick something like so101-mlflow-<suffix>."
  type        = string
}

variable "allowed_cidr" {
  description = "The only CIDR allowed to reach the server, e.g. the lab's public IP as a /32. There is no login on MLflow itself."
  type        = string
  validation {
    condition     = can(cidrnetmask(var.allowed_cidr))
    error_message = "allowed_cidr must be an IPv4 CIDR such as 203.0.113.7/32."
  }
}

variable "db_instance_class" {
  description = "RDS instance class. db.t4g.micro is the smallest burstable class and the only recurring cost of note."
  type        = string
  default     = "db.t4g.micro"
}

variable "cpu_architecture" {
  description = "Fargate CPU architecture. ARM64 runs the image a Mac builds natively and is about 20 percent cheaper than X86_64."
  type        = string
  default     = "ARM64"
  validation {
    condition     = contains(["ARM64", "X86_64"], var.cpu_architecture)
    error_message = "cpu_architecture must be ARM64 or X86_64."
  }
}

variable "desired_count" {
  description = "Number of server tasks: 1 to run, 0 to pause the Fargate charge while keeping the database and bucket."
  type        = number
  default     = 1
}
