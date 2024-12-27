variable "project_id" {
    description = "Google Cloud project ID"
    type = string    
}

variable "region" {
    description = "Region to deploy Google Cloud resources"
    type = string
}

variable "cpu_limit" {
    description = "CPU limit for cloud run serverless infra. E.g. 2"
    type = string     
}

variable "memory_limit" {
  description = "Memory limit for cloud run serverless infra in Gi. E.g. 4Gi"
  type        = string
}

variable "registry_name" {
    description = "Name of docker artifact registry"
    type = string 
}

variable "app_name" {
    description = "Name of the app. E.g. backend, frontend etc."
    type = string 
}