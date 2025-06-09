docker build -t gcr.io/cv-web-app-425302/cv-web-app:latest .
docker push gcr.io/cv-web-app-425302/cv-web-app:latest
gcloud run deploy cv-web-app --image gcr.io/cv-web-app-425302/cv-web-app:latest --region us-central1 --allow-unauthenticated --memory 4G --cpu 1 --max-instances 2
