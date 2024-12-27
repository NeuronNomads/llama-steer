module "backend" {
  source     = "../../modules/cloud_run"
  project_id = var.project_id
  region = var.region
  cpu_limit = var.backend_cpu_limit
  memory_limit = var.backend_memory_limit
  registry_name = var.backend_registry_name
  app_name = var.backend_app_name
}