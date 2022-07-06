# Deploy GNN models with TigerGraph on KServe 
1. Train your GCN model and save it to Google cloud storage using `gcn_node_classification.ipynb`
2. Define custom model server image by defining the Dockerfile
3. Use the build.sh to determine the python application, install the dependencies from the requirements.txt file, build, and push the custom model server image 
    - chmod +x build.sh
    - ./build.sh
5. Deploy the Custom Predictor on KServe using `Kserve_deployment.ipynb`
6. Perform Inference by sending an inference request to the deployed model in order to get predictions. We assume that you running it in your Kubeflow cluster and will use the internal URL of the InferenceService.
7. Delete InferenceService when you are done with this service.
