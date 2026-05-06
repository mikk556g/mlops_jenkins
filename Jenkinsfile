pipeline {
    agent { label 'GPU Worker 1'}

    environment {
        IMAGE_NAME = "lam_mlops_image"
        COMMIT_HASH = "${GIT_COMMIT.take(7)}"
        MLFLOW_TRACKING_URI = "http://172.24.198.42:5050"
        DOCKER_REGISTRY = "172.24.198.42:5000"
    }

    stages {

        stage('Build Docker Image') {
            steps {
                echo "Building docker image"
                // Installs dependecies and requirements inside the Dockerfile"
                sh "docker build -t ${IMAGE_NAME}:${COMMIT_HASH} ."
            }
        }

        stage('Pull Dataset') {
            steps {
                echo "Pulling dataset"
                sh "docker run --rm -v \$(pwd):/project ${IMAGE_NAME}:${COMMIT_HASH} dvc pull"
            }
        }

        stage('Run Unit Tests') {
            steps {
                echo "Running unit tests"
                sh "docker run --rm ${IMAGE_NAME}:${COMMIT_HASH} python3 -m pytest"
            }
        }

        stage('Train Model') {
            steps {
                echo "Training model"
                sh """
                    docker run --rm --gpus 1 \
                    -v \$(pwd)/data:/project/data \
                    ${IMAGE_NAME}:${COMMIT_HASH} \
                    python3 train.py --mlflow-uri ${MLFLOW_TRACKING_URI}
                """
            }
        }

        stage('Evaluate Model') {
            steps {
                echo "Evaluating model"
                sh """
                    docker run --rm --gpus 1 \
                    -v \$(pwd)/data:/project/data \
                    ${IMAGE_NAME}:${COMMIT_HASH} \
                    python3 test.py --mlflow-uri ${MLFLOW_TRACKING_URI}
                """
            }
        }

        stage('Export to ONNX and Quantize Model') {
            steps {
                echo "Exporting to ONNX and quantizing model to INT8"
                sh "mkdir -p \$(pwd)/models"
                sh """
                    docker run --rm --gpus 1 \
                    -v \$(pwd)/models:/project/models \
                    ${IMAGE_NAME}:${COMMIT_HASH} \
                    bash -c "python3 export_onnx.py --mlflow-uri ${MLFLOW_TRACKING_URI} && \
                             trtexec --onnx=/project/models/model.onnx --saveEngine=/project/models/model_fp16.engine --fp16 && \
                             python3 log_engine.py --mlflow-uri ${MLFLOW_TRACKING_URI} --engine /project/models/model_fp16.engine"
                """
            }
        }

        stage('Push Docker Image') {
            steps {
                echo "Pushing docker image to registry"
                sh "docker tag ${IMAGE_NAME}:${COMMIT_HASH} ${DOCKER_REGISTRY}/${IMAGE_NAME}:${COMMIT_HASH}"
                sh "docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${COMMIT_HASH}"
            }
        }
    }

    post {
        always {
            sh "docker rmi ${IMAGE_NAME}:${COMMIT_HASH} || true"
            sh "docker rmi ${DOCKER_REGISTRY}/${IMAGE_NAME}:${COMMIT_HASH} || true"
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
