# so101 MLflow on AWS: the compose stack's shape (infra/docker-compose.yml) as managed services. Nothing here runs or costs
# anything until `terraform apply`; see docs/mlops/infra.md for the mapping, the monthly cost and what is deliberately left out.
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = { project = var.project, managed_by = "terraform" }
  }
}
