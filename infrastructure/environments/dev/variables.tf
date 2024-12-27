variable "project_id" {
    description = "Google Cloud project ID"
    type = string    
}

variable "region" {
    description = "Region to deploy Google Cloud resources"
    type = string
}

variable "env" {
    description = "Environment to deploy Google Cloud resources"
    type = string
}