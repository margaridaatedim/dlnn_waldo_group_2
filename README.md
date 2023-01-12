# Deep Learning Neural Networks (Group 2) - Where's Waldo? 

## Introduction 

In this project, we present a deep learning approach for detecting the character Waldo in images. This project is the result of an assigned project for the Deep Learning Neural Networks course at the post-graduation in Enterprise Analytics and Data Science of NOVAIMS university. The goal of the project is to develop a model that can classify images as containing or not containing Waldo. To train the model, we compiled a dataset of scenarios with and without Waldo and used a convolutional neural network (CNN) approach to learn features from the images. We built a set of different models and, upon testing, we selected the best-performing one based on F1-score results. We then applied hyperparameter tuning techniques, such as Dropout, L2 Regularization, and Batch Normalization, to further improve the performance of the model. Our final goal was to build a classifier with an F1-Score higher than 50. In addition, we also applied Transfer Learning by using a pre-trained model to our dataset. Lastly, we used object detection techniques to find Waldo in a set of sample images.

The team behind this project is António Martins, Clara Barreto, Larissa de Lucia, Lúcio Roque, and Margarida Tedim.

## Repository files and usage 

### CNN Binary Classification

#### main file: Binary_Classification_Wally.ipynb 

Where we developed an CNN approach to classify wally images. The following phases were included in this file:

**Data Exploration: In this phase, we got acquainted with the dataset structure and the issues that needed fixing.
**Preprocessing: In this phase, we did the data transformations required for modelling, in which we built various models and identified the most suitable architecture based on its F1-Score.
**Modeling: In this phase, we built various models and identified the most suitable architecture based on its F1-Score.
**Hyper-parameter tuning: As an attempt to improve the selected model, we tested Hyper-parameter tuning techniques to improve its performance.
**Transfer Learning: As an attempt to improve the selected model, we tested Transfer Learning techniques to improve its performance.

#### Data Acquisition

##### In this work, we aimed to acquire data from Kaggle to use in Google Colab. 

Here are the steps we followed:

We installed the Kaggle Python package by running the command !pip install kaggle in our Google Colab notebook.
We logged in to the Kaggle website and accessed our account page. In the "API" section, we clicked on the "Create New API Token" button, which downloaded a JSON file with our API credentials.
The kaggle.json file downloaded has to be uploaded in one of the first cells of the notebook. The following steps to download data from Kaggle are in the notebook.

##### wallymerge.zip: The data used to increase Waldo images and try to enhance unbalanced classes.

As a first approach, we decided to proceed with  the Kaggle 128x128 colour images dataset for the model development considering had a better class distribution and more data.
Considering we still had 2% of “waldo” images we decided to enrich our primary dataset with the 64x64 and 256x256 colour images from the original dataset and with images we took from the original books that is cointained in this wallymerge.zip file.

### Object Detection

#### Training_Object_Detection_YOLOv3_Model.ipynb

 we used the YOLO algorithm (“You Only Look Once”), which is real-time object detection algorithm, to train a model for detecting Waldo in images.

#### ObjectDetection_VisualizingImages.ipynb 

In this file, we used the YOLO3 model trained in the previous file to predict the location of Waldo and define bounding boxes around him. The data used for prediction is in a public folder of Google Drive and you have to access the folder in your notebook. You may need to adjust the file path slightly depending on where you are running your notebook. The link to the folder is: https://drive.google.com/drive/folders/1PqYWwEwyzbhACvMI-hQhUOqJh0h27Vw4?usp=sharing

#### ObjectDetection_JustForFun.ipynb 

As a fun demonstration of the capabilities of object detection, in this file we adapted a model to detect and predict the presence of the characters Professor Illya or Mafalda as Waldo. This implementation takes a background image and randomly places one of the Professors to train and test the model. Unfortunately, due to time constraints, we were not able to apply this approach to the actual dataset. The data used for this model is in a waldoprofessors.zip file.

## Conclusion

In this project, we successfully developed a deep learning model for detecting Waldo in images using CNN and object detection techniques. We also demonstrated the potential for further experimentation and application of these techniques to other similar problems. We encourage readers to use and build upon our work for their own projects and research.

Regarding the model results, we achieved an F1-Score higher than 50, which fulfilled our goal for this project. The process of implementing and trying to improve the performance of the model was a valuable learning experience for our team. We hope that our approach and methods can be useful for others in the data science community.

## Acknowledgments

We would like to extend our sincere gratitude to the professors of the Deep Learning Neural Networks course at the post-graduation in Enterprise Analytics and Data Science of NOVAIMS university, especially to Professor Mafalda, for their invaluable guidance and support throughout this project. Their expertise and knowledge have been instrumental in shaping this project.

We would also like to thank the data science community for sharing their knowledge and resources through various videos, documentations, and forums. Without their contributions, this project would not have been possible. Their willingness to share their expertise and insights has been a valuable resource for our team throughout this project.

