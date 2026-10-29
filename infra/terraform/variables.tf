variable "project_id" {
  description = "GCP project to deploy into"
  type        = string
}

variable "region" {
  description = "Region for the Autopilot cluster and the registry"
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  type    = string
  default = "brainct"
}

variable "deletion_protection" {
  description = "Keep false for this demo so `terraform destroy` works in one step"
  type        = bool
  default     = false
}
