# llama-steer

Steering llama3-8B-Instruct using Sparse Autoencoder latents.

## Setup

Before setting up your environment, make sure you have the following installed (TODO: Add versions):
1. `gcloud` 
2. `poetry`
3. `terraform` 

### Backend

From the backend directory, 

1. Run `make auth` to authenticate to the `llama-steer` google cloud project.
2. Run `make install` to setup your local environment. 