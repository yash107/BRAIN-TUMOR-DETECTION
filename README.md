# Brain Tumor and Alzheimer's Disease Detection using Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) for the detection of brain tumors and the diagnosis of Alzheimer's disease from medical imaging data. The CNN model is trained to classify MRI images into two categories: images with brain tumors or signs of Alzheimer's disease and images without these conditions.
### Dataset

The dataset used for training and evaluation consists of MRI images of the brain, obtained from Kaggle. It comprises a total of 3264 images, with a subset containing brain tumor data. The dataset is preprocessed and split into training, validation, and test sets to facilitate model training and evaluation.

### Model Architecture
The CNN model architecture used for brain tumor and Alzheimer's disease detection is as follows:

1. Input layer: Accepts MRI images with dimensions (150,150,3).</br></br>
2. Convolutional layers: Multiple convolutional layers with varying filter sizes and activation functions (ReLU) are used to extract features from the input images.</br></br>
3. MaxPooling layers: MaxPooling layers are employed to downsample the feature maps, retaining the most important information.</br></br>
4. Dropout layers: Dropout layers are included for regularization to prevent overfitting.</br></br>
5. Flatten layer: Flattens the 2D feature maps into a 1D vector.</br></br>
6. Fully connected layers: Dense layers are utilized for classification, with ReLU activation functions.</br></br>
7. Output layer: The final output layer employs the softmax activation function to classify MRI images into tumor/Alzheimer's disease or non-tumor/non-Alzheimer's disease classes.</br></br>
### Training
The model is trained using the training dataset for 20 epochs with data augmentation techniques to increase the diversity of the training samples and improve model generalization. The training process involves optimizing the categorical cross-entropy loss function using the Adam optimizer.

### Evaluation
The trained model is evaluated on the validation dataset to assess its performance in terms of accuracy, precision, recall, and F1-score. Additionally, the model's performance is evaluated on an independent test dataset to ensure unbiased evaluation.

### Results
The model achieves an accuracy of 93.7% on the test dataset, demonstrating its effectiveness in detecting brain tumors and diagnosing Alzheimer's disease from MRI images.

### Streamlit Integration
The model is integrated with a Streamlit frontend to provide an interactive web interface for users to upload MRI images and obtain predictions regarding the presence of brain tumors or signs of Alzheimer's disease. The Streamlit app allows users to visualize model predictions and provides a user-friendly experience for medical professionals and patients.

### Usage
To utilize the trained model and Streamlit app for brain tumor and Alzheimer's disease detection:
Run the Streamlit app using the command streamlit run stream.py.
Upload MRI images through the web interface and obtain predictions regarding brain tumor or Alzheimer's disease diagnosis.

### Conclusion
In conclusion, the developed CNN model, achieving an accuracy of 93.7%, coupled with the Streamlit frontend, provides an effective tool for detecting brain tumors and diagnosing Alzheimer's disease from MRI images. The integration of a user-friendly interface enhances accessibility and usability, making the model applicable in clinical settings for improved patient care and outcomes.
