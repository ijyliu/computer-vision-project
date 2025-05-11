# computer-vision-project

Isaac Liu, Mayank Sethi, Gaurav Sharan Srivastava, Ashutosh Tiwari

This project tackles an image classification task derived from the Stanford Cars Dataset.

In Phase 1, we preprocessed and resized >8,000 images (blurring with convolutions when appropriate), and constructed classical (gradient, color, texture) and neural (convolutional neural net, vision transformer) embeddings (additionally visualized with tSNE). We then deployed 4 different classfiers (Logistic Regression, SVM, XGBoost, Random Forest) on several variations of the data with no or variable amounts of dimensionality reduction applied via PCA, and performed comprehensive grid search for hyperparameters with 5-fold cross validation. We were able to achieve >91% test accuracy with our best classifier (with further efficiency improvements possible with minimal cost to accuracy) - for details, see this [report](https://docs.google.com/document/d/1Hm4_qpn-m_Z5ploa43l7hY6AJFA5l4dOiKIlIt_PWf4/edit#heading=h.rqcihi4c7zuc) and [presentation](https://docs.google.com/presentation/d/1_uGFRyL-al_7lUX1pT7Er69vDq8aFttjd4ZpQKmEFmc/edit?usp=sharing).

In Phase 2, Isaac finetuned a mid-sized ResNet convolutional neural network on the data in PyTorch (>93% accuracy) and deployed a [web app](https://ijyliu.github.io/cv-web-app-loading-page/loading.html) with Flask, Docker, and Google Cloud - you can upload your own image.

## Technologies (not exhaustive!)

- Python
  - Scikit-Image
  - OpenCV
  - Transformers
  - Sklearn
  - XGBoost
  - PyTorch
  - Flask
  - Pandas
  - Conda
- Bash and the Slurm Cluster Resource Manager for CPUs and GPUs
- Docker
- Google Cloud
