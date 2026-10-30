terraform {
  required_version = ">= 1.6"
  # State is local (gitignored) for this one-person demo. For a team, keep it in GCS:
  #   backend "gcs" { bucket = "<state-bucket>"  prefix = "brainct" }
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
