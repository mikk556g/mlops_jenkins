# MLOps
MLOps Projekt

This project build upon a facial emotion recognition AI model...



Mangler:
- Pipeline trigger based on a new commit (webhook)
- Implementation of automatic creation of model card at the end of train.py
- Deployment stage, with deploy.py script
- Enable branch protection, and automatically merge new features to main,
  (e.g. merges must require a successful complete run of your MLOps pipeline). You can handle this with Jenkins.
- Lige nu bliver den færdig-trænede model og dens artifakter også gemt inde i
  det docker image der bliver sendt til docker registry serveren - dette er nok ikke
  optimalt og skal måske ændres.
