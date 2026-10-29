# GKE Autopilot + Artifact Registry for the BrainCT API. Pay-per-pod: nothing runs (and little
# is billed) until the Helm release is installed. Tear down with `terraform destroy`.

locals {
  services = [
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
    "compute.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudbuild.googleapis.com",
  ]
  labels = { app = "brainct", managed-by = "terraform" }
}

resource "google_project_service" "apis" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "brainct"
  format        = "DOCKER"
  description   = "BrainCT API and load-test images"
  labels        = local.labels

  # Keep the last 5 versions; older untagged images are removed automatically.
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }
  depends_on = [google_project_service.apis]
}

# Nodes run as a dedicated service account with only what they need: pull images, write
# logs and metrics. (The default compute SA has project-wide Editor.)
resource "google_service_account" "nodes" {
  account_id   = "${var.cluster_name}-nodes"
  display_name = "BrainCT GKE nodes"
}

resource "google_project_iam_member" "nodes" {
  for_each = toset([
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.nodes.email}"
}

resource "google_container_cluster" "autopilot" {
  name                = var.cluster_name
  location            = var.region
  enable_autopilot    = true
  deletion_protection = var.deletion_protection
  resource_labels     = local.labels

  release_channel {
    channel = "REGULAR"
  }

  cluster_autoscaling {
    auto_provisioning_defaults {
      service_account = google_service_account.nodes.email
      oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    }
  }

  # Kubernetes NetworkPolicy enforcement (the chart ships one) comes with Dataplane V2.
  datapath_provider = "ADVANCED_DATAPATH"

  depends_on = [google_project_service.apis, google_project_iam_member.nodes]
}
