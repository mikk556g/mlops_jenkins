pipeline {
    agent any  // run on the Jenkins server container

    triggers {
        pollSCM('H/* * 2 * *')  // polls every 2 days
    }

    environment {
        IMAGE_NAME = "lam_mlops_image"
        COMMIT_HASH = "${GIT_COMMIT.take(7)}"
        MLFLOW_TRACKING_URI = "http://172.24.198.42:5050"  // local MLflow logs
        DOCKER_REGISTRY_URI = 172.24.198.42:5000
    }

    stages {

        stage('Build Docker Image') {
            steps {
                // Installs dependecies and requirements inside the Dockerfile"
                sh "docker build -t ${IMAGE_NAME}:${COMMIT_HASH} ."
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh "docker run --rm ${IMAGE_NAME}:${COMMIT_HASH} python3 -m pytest"
            }
        }

        stage('Train Model') {
            steps {
                sh "docker run --rm ${IMAGE_NAME}:${COMMIT_HASH} python3 train.py --mlflow-uri ${MLFLOW_TRACKING_URI}"
            }
        }

        stage('Evaluate Model') {
            steps {
                sh "docker run --rm ${IMAGE_NAME}:${COMMIT_HASH} python3 test.py --mlflow-uri ${MLFLOW_TRACKING_URI}"
            }
        }

        stage('Export to ONNX') {
            steps (
                sh "docker run --rm ${IMAGE_NAME}:${COMMIT_HASH} python3 export_onnx.py --mlflow-uri ${MLFLOW_TRACKING_URI}"
            )
        }

        stage('Push Docker Image') {
            steps {
                sh "docker tag ${IMAGE_NAME}:${COMMIT_HASH} ${DOCKER_REGISTRY_URI}/${IMAGE_NAME}:${COMMIT_HASH}"
                sh "docker push ${DOCKER_REGISTRY_URI}/${IMAGE_NAME}:${COMMIT_HASH}"
            }
        }

        stage('Deploy Model') {
            steps {
                sh "docker run --rm ${IMAGE_NAME}:${COMMIT_HASH} python3 deploy.py --mlflow-uri ${MLFLOW_TRACKING_URI}"
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished!'
        }

        success {
            archiveArtifacts artifacts: '**/model_cards/*.md', allowEmptyArchive: true
            echo 'Pipeline succeeded!'
        }

        failure {
            echo 'Pipeline failed. Check logs!'
        }
    }
}
