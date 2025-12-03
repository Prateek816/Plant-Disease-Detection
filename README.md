# Plant-Disease-Detection
A deep-learning system that classifies plant leaf images into 38 disease/healthy classes using a custom Convolutional Neural Network (CNN)
Dataset used:
New Plant Diseases Dataset (Kaggle)
https://www.kaggle.com/datasets/vipooooool/new-plant-diseases-dataset/data

 Features
Multi-class classification (38 classes) using TensorFlow/Keras CNN.
Accuracy visualization, confusion matrix, and evaluation metrics.
Leaf disease severity estimation:
Segments infected regions
Computes infection ratio = diseased area ÷ leaf area
Maps to severity levels: Mild / Moderate / Severe
Streamlit web app (website.py) for easy image upload & prediction.
Single-image inference script (model.py).
Trained model included (trained_model.h5) for quick testing.



plant-disease-recognition/
├── train_model.py        # training script (CNN + evaluation)
├── model.py              # single-image classification
├── website.py            # saved model
├── history.json          # training history
├── home_page.jpeg        # homepage image for Streamlit
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
└── README.md



Download the dataset from Kaggle and arrange like this:
data/train/<38 class folders>
data/valid/<38 class folders>
data/test/<images>


python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
.\venv\Scripts\activate         # Windows

pip install -r requirements.txt



Training the Model -

python train_model.py

The script:
Loads images using image_dataset_from_directory
Builds a CNN with 5 convolution blocks
Trains for 10 epochs
Saves:
trained_model.h5
history.json
Accuracy plots



Run Single-Image Prediction
python model.py
This script:
Loads trained_model.h5
Preprocesses a single leaf image
Outputs predicted class name


🌐 Running the Streamlit Web App
streamlit run website.py
Features:
Upload any plant leaf image
Model predicts the disease class instantly
Clean, simple UI for users


📊 Evaluation
Includes:
Accuracy curves (train vs validation)
Classification report
Confusion matrix heatmap
Metrics saved in history.json

