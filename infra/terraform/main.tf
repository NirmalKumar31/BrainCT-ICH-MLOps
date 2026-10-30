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

  # Tags cannot be moved: a rebuilt image needs a new tag (<version>-<git sha>), so nodes
  # with a cached copy (imagePullPolicy IfNotPresent) can never run a stale model.
  docker_config {
    immutable_tags = true
  }

  # A KEEP policy alone deletes nothing: Artifact Registry removes only what a DELETE policy
  # matches and no KEEP policy protects. Delete versions older than 30 days, keep the last 5.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      older_than = "2592000s"
    }
  }
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

# Cloud Build pushes images as its own service account, not the default one (which has broad
# project roles): write to this one repository, write build logs.
resource "google_service_account" "build" {
  account_id   = "${var.cluster_name}-build"
  display_name = "BrainCT image builds"
}

resource "google_artifact_registry_repository_iam_member" "build_push" {
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.build.email}"
}

resource "google_project_iam_member" "build_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.build.email}"
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

  # The control plane has a public endpoint; restrict who can reach it (e.g. your IP/32).
  # Empty = unrestricted, so a first `apply` works from anywhere.
  dynamic "master_authorized_networks_config" {
    for_each = length(var.authorized_networks) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.authorized_networks
        content {
          cidr_block   = cidr_blocks.value
          display_name = "authorized"
        }
      }
    }
  }

  depends_on = [google_project_service.apis, google_project_iam_member.nodes]
}
