output "registry" {
  description = "Push images here: <registry>/brainct-api:<tag>"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "build_service_account" {
  description = "gcloud builds submit --service-account=projects/<project>/serviceAccounts/<this>"
  value       = google_service_account.build.email
}

output "get_credentials" {
  value = "gcloud container clusters get-credentials ${google_container_cluster.autopilot.name} --region ${var.region} --project ${var.project_id}"
}
