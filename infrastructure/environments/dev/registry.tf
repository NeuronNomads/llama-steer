module "docker_artifact_registry" {
  source     = "github.com/GoogleCloudPlatform/cloud-foundation-fabric//modules/artifact-registry?ref=v36.0.0"
  project_id = var.project_id
  location   = var.region
  name       = "backend"
  format = {
    docker = {
      standard = {
        immutable_tags = false
      }
    }
  }
}