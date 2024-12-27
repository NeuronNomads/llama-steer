// Every cloud run service has an accompanied service account (SA). This SA will be used
// for allocating appropriate permissions.
module "cloud_run_sa" {  
  source     = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/iam-service-account?ref=v36.0.0"
  project_id = var.project_id
  name       = "${var.app_name}-sa"

  # non-authoritative roles granted *to* the service accounts on other resources
  iam_project_roles = {
    "${var.project_id}" = [
      "roles/logging.logWriter",            # Required to write logs for debugging 
      "roles/storage.objectUser",           # Required to read models from buckets
      "roles/iam.serviceAccountUser",       # Required to attach the SA to cloud run       
      "roles/secretmanager.secretAccessor", # Required to access secrets
    ]
  }
}

// Cloud run is google's managed serverless product. It allows an image to be specified, and cloud run will
// autoscale the container for you depending on the resource requirements.
module "cloud_run" {  
  source     = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/cloud-run-v2?ref=v36.0.0"
  project_id = var.project_id
  region     = var.region
  name       = "${var.app_name}-cloud-run"  
  containers = {
    container-1 = {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.registry_name}/${var.project_id}:latest"
      
      resources = {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
      }
    }
  }  
  service_account = module.cloud_run_sa.email
  // Ensure the cloud run service account is created prior to the creation of the cloud run resource
  depends_on = [module.cloud_run_sa]
}