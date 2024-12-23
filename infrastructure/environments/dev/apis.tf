module "apis" {
  source     = "../../modules/apis"
  project_id = var.project_id
  api_name   = each.key
  for_each = toset([
    "serviceusage", # Must be enabled to apply this with terraform
    "cloudresourcemanager",
    "iam",
    "storage",    
    "logging",
    "run",
    "secretmanager",    
    "billingbudgets",
    "artifactregistry",
  ])
}