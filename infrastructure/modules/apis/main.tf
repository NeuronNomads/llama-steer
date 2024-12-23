resource "google_project_service" "api" {
  project = var.project_id
  service = "${var.api_name}.googleapis.com"
  timeouts {
    create = "30m"
    update = "40m"
  }
  disable_dependent_services = false
  # don't disable serviceusage API, as it must be enabled for this module to work
  disable_on_destroy = var.api_name == "serviceusage" ? false : true
}

# Enabling APIs are eventually consistent, so wait 2min
resource "time_sleep" "wait_for_api" {
  depends_on      = [google_project_service.api]
  create_duration = "120s"
}