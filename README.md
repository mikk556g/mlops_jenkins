# MLOps
MLOps Projekt

This project build upon a facial emotion recognition AI model...


If your registry is plain HTTP (not HTTPS), you need to tell Docker to allow it as an insecure registry on the Jenkins host. Add this to /etc/docker/daemon.json:

{
  "insecure-registries": ["<IP>:<PORT>"]
}

Then restart Docker: sudo systemctl restart docker



stage('Deploy Model') {
    steps {
        sh "docker run --rm ${IMAGE_NAME}:${COMMIT_HASH} python3 deploy.py --mlflow-uri ${MLFLOW_TRACKING_URI}"
    }
}
