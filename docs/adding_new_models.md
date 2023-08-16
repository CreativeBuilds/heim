# Adding new models

Supporting new models is similar to how it's done in HELM. 
Refer to the HELM documentation [here](https://crfm-helm.readthedocs.io/en/latest/adding_new_models/).

The [HuggingFaceDiffusersClient](https://github.com/stanford-crfm/heim/blob/main/src/helm/proxy/clients/image_generation/huggingface_diffusers_client.py) 
is a good example of a client that hosts multiple text-to-image models.
