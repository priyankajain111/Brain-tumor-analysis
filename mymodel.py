import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


from skimage.util import montage 
from skimage.transform import rotate

from sklearn.preprocessing import MinMaxScaler
from sklearn import tree

import random
import cv2


import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import img_to_array

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torchvision import models
# from torchvision import transforms

# from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Activation


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC,SVR

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc, accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize


import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import sklearn.metrics as metrics

from sklearn.multiclass import OneVsRestClassifier

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve,auc

import seaborn as sns
from tqdm import tqdm

#VGG
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from skimage.util import montage
from skimage.transform import rotate

from sklearn.preprocessing import MinMaxScaler
from sklearn import tree

import random
import cv2

from keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
from keras import Input
from pickle import dump

import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, GlobalAveragePooling2D, Flatten

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, SVR

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.multiclass import OneVsRestClassifier

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  ConfusionMatrixDisplay,classification_report
from scikitplot.metrics import confusion_matrix

import matplotlib as mpl

import torch
import numpy as np
from tqdm import tqdm
  

df_ids = pd.read_csv("D:/flask/uploads/my_data.csv")
    
class ModelFeatureExtractor:
    def __init__(self, model, modalities):
        self.model_name = model
        self.modalities = modalities
        self.features_avg_alex = []  # Empty list to store average features extracted by AlexNet
        self.features_std_alex = []  # Empty list to store standard deviation of features extracted by AlexNet
        self.features_avg_vgg = []  # Empty list to store average features extracted by VGG
        self.features_std_vgg = []  # Empty list to store standard deviation of features extracted by VGG

        if model == "AlexNet":
            # Load AlexNet model with pre-trained weights
            model = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
            
            # Remove last two layers of AlexNet (classifier layers) to get feature extraction layers
            alexnet = torch.nn.Sequential(*(list(model.children())[:-2]))
            
            # Apply average pooling with kernel size 6x6
            alexnet = nn.Sequential(alexnet, nn.AvgPool2d(6))
            
            # Flatten the output feature maps
            alexnet = nn.Sequential(alexnet, nn.Flatten())
            
            self.model = alexnet
        else:
            # Load VGG16 model with pre-trained weights
            vgg_model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
            
            # Remove the last layer of VGG16 to get feature extraction layers
            features_layer = nn.Sequential(*list(vgg_model.features.children())[:-1])
            
            # Apply global average pooling to reduce spatial dimensions to 1x1
            global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Flatten the output feature maps
            vggnet = nn.Sequential(features_layer, global_avg_pool, nn.Flatten())
            
            self.model = vggnet
            
    def interpolate(self, new_image_data):
        h, w = 224, 224  # Target height and width for interpolation
        dim = (w, h)  # Tuple representing the target dimensions
        final_image = cv2.resize(new_image_data, dim, interpolation=cv2.INTER_CUBIC)
        # Resize the new image data to the target dimensions using bicubic interpolation

        return final_image

    
    def convert(self,image,model):
        if model == "AlexNet":
            # Resize the image to 224x224
            image_new = cv2.resize(image, (224, 224))

            # Normalize the image to values between 0 and 1
            image_new = (image_new - np.min(image_new)) / (np.max(image_new) - np.min(image_new))

            # Convert the image to RGB by stacking the grayscale image 3 times
            image_new = np.stack((image_new,)*3, axis=-1)

            # Reshape the image to (1, 3, 224, 224) so that it suits AlexNet
            image_new = np.expand_dims(np.transpose(image_new, (2, 0, 1)), axis=0)

            return image_new
        elif model == "VGGNet":
            # Resize the image to 224x224
            image_new = cv2.resize(image, (224, 224))

            # Normalize the image to values between 0 and 1
            image_new = (image_new - np.min(image_new)) / (np.max(image_new) - np.min(image_new))

            # Convert the image to RGB by stacking the grayscale image 3 times
            image_new = np.stack((image_new,)*3, axis=-1)

            # Reshape the image to (1, 224, 224, 3) so that it suits VGG
            image_new = np.expand_dims(image_new, axis=0)

            return image_new

    def slice_range(self, image_data_seg):
        slices = 155  # Total number of slices
        start = 0  # Initial value for the start slice
        end = 154  # Initial value for the end slice

        # Find the start slice
        for n_slice in range(slices):
            image = image_data_seg[:, :, n_slice]

            # Check if the image contains specific values (1, 2, or 4) and has more than 300 nonzero elements
            if (1 in image or 2 in image or 4 in image) and np.count_nonzero(image) > 300:
                start = n_slice
                break

        # Find the end slice
        for n_slice in range(slices-1, -1, -1):
            image = image_data_seg[:, :, n_slice]

            # Check if the image contains specific values (1, 2, or 4) and has more than 300 nonzero elements
            if (1 in image or 2 in image or 4 in image) and np.count_nonzero(image) > 300:
                end = n_slice
                break

        return start, end

    def bbox(self, img):
        # Find the rows and columns that contain nonzero elements
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)

        # Find the minimum and maximum row indices that contain nonzero elements
        rmin, rmax = np.where(rows)[0][[0, -1]]

        # Find the minimum and maximum column indices that contain nonzero elements
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax


    def preprocess_image_slices(self, patient_id, modality, model):
        # Add your preprocess_image_slices logic here
        
        # Load image data
        image_path = f'D:/flask/uploads/{patient_id}_{modality}.nii.gz'
        image_data = nib.load(image_path).get_fdata()

        # Load segmentation data
        seg_path = f'D:/flask/uploads/{patient_id}_automated_approx_segm.nii.gz'
        image_data_seg = nib.load(seg_path).get_fdata()

        # Compute bounding box and slice range
        rmin, rmax, cmin, cmax = self.bbox(image_data_seg)
        start, end = self.slice_range(image_data_seg)
        size = end - start + 1

        # Preprocess image slices
        if model =="AlexNet":
            x = np.zeros((size, 3, 224, 224), dtype=np.float32)
        elif model=="VGGNet":
            x = np.zeros((size,224,224,3),dtype=np.float32)

        for i in range(start, end+1):
            # Extract a 2D slice from the 3D image volume
            new_image_data = image_data[rmin:rmax+1, cmin:cmax+1, i]

            # Resize the image to 224x224 and convert to RGB format
            final_image = self.interpolate(new_image_data)
            final_image = self.convert(final_image,model)

            # Add the processed image slice to the input tensor
            x[i-start] = final_image

        return x

    def extract_features_alex(self, df_ids):
        for ind, patient_id in tqdm(enumerate(df_ids['ID']), total=len(df_ids['ID'])): 
            features_row_avg_alex = []  # Empty list to store average features for the current patient
            features_row_std_alex = []  # Empty list to store standard deviation of features for the current patient

            for modality in self.modalities:
                # Preprocess image slices
                x = self.preprocess_image_slices(patient_id, modality, self.model_name)

                # Extract features using pre-trained AlexNet
                x_torch = torch.from_numpy(x)

                features_torch = self.model(x_torch)
                features_numpy = features_torch.detach().numpy()

                # Concatenate the average and standard deviation of features to the current patient's feature lists
                features_row_avg_alex = np.concatenate([features_row_avg_alex, np.average(features_numpy, axis=0)], axis=0)
                features_row_std_alex = np.concatenate([features_row_std_alex, np.std(features_numpy, axis=0)], axis=0)

            # Append the current patient's feature lists to the overall feature lists
            self.features_avg_alex.append(features_row_avg_alex)
            self.features_std_alex.append(features_row_std_alex)


    def extract_features_vgg(self, df_ids):
        for ind, patient_id in tqdm(enumerate(df_ids['ID']), total=len(df_ids['ID'])):
            features_row_avg_vgg = []  # Empty list to store average features for the current patient
            features_row_std_vgg = []  # Empty list to store standard deviation of features for the current patient
            for modality in modalities:
                # Preprocess image slices
                x = self.preprocess_image_slices(patient_id, modality, self.model_name)

                x_torch = torch.from_numpy(x)
                # Transpose the dimensions of the input tensor
                x_torch = torch.from_numpy(x).permute(0, 3, 1, 2)

                # Extract features using pre-trained VGG16
                with torch.no_grad():
                    features_tensor = self.model(x_torch)
                features_numpy = features_tensor.numpy()

                # Concatenate the average and standard deviation of features to the current patient's feature lists
                features_row_avg_vgg = np.concatenate([features_row_avg_vgg, np.average(features_numpy, axis=0)], axis=0)
                features_row_std_vgg = np.concatenate([features_row_std_vgg, np.std(features_numpy, axis=0)], axis=0)

            # Append the current patient's feature lists to the overall feature lists
            self.features_avg_vgg.append(features_row_avg_vgg)
            self.features_std_vgg.append(features_row_std_vgg)
modalities = ["FLAIR", "T1GD", "T1", "T2"]
alexnet_extractor = ModelFeatureExtractor("AlexNet", modalities)
alexnet_extractor.extract_features_alex(df_ids)
vggnet_extractor = ModelFeatureExtractor("VGGNet", modalities)
vggnet_extractor.extract_features_vgg(df_ids)
panda_features_avg_alex = pd.DataFrame(data = alexnet_extractor.features_avg_alex)
panda_features_std_alex = pd.DataFrame(data = alexnet_extractor.features_std_alex)
panda_features_avg_vgg = pd.DataFrame(data = vggnet_extractor.features_avg_vgg)
panda_features_std_vgg = pd.DataFrame(data = vggnet_extractor.features_std_vgg)
print(panda_features_avg_alex.shape,panda_features_avg_vgg.shape)


def rename():
    modalities = ["FLAIR", "T1GD", "T1", "T2"]
    n_features_alex = 256
    n_features_vgg = 512
    columns_avg_alex = [f"{modality}_AlexNet_Avg_F{i}" for modality in modalities for i in range(n_features_alex)]
    columns_std_alex  = [f"{modality}_AlexNet_Std_F{i}" for modality in modalities for i in range(n_features_alex)]

    columns_avg_vgg = [f"{modality}_VGGNet_Avg_F{i}" for modality in modalities for i in range(n_features_vgg)]
    columns_std_vgg  = [f"{modality}_VGGNet_Std_F{i}" for modality in modalities for i in range(n_features_vgg)]

    panda_features_avg_alex.columns = columns_avg_alex
    panda_features_std_alex.columns = columns_std_alex

    panda_features_avg_vgg.columns = columns_avg_vgg
    panda_features_std_vgg.columns = columns_std_vgg



def concatenate():
    final_alex_features = pd.concat([panda_features_avg_alex, panda_features_std_alex], axis=1)
    final_vgg_features = pd.concat([panda_features_avg_vgg, panda_features_std_vgg], axis=1)
    final_alex_features.to_csv("D:/flask/uploads/final_alex_features.csv",index=False,header = final_alex_features.columns)
    final_vgg_features.to_csv("D:/flask/uploads/final_vgg_features.csv",index=False,header = final_vgg_features.columns)
    df1 = pd.read_csv("D:/flask/uploads/final_alex_features.csv")
    df2 = pd.read_csv("D:/flask/uploads/final_vgg_features.csv")
    merged_df = pd.concat([df1, df2], axis=1)
    merged_df.to_csv("D:/flask/uploads/final_features.csv", index=False)
    print(merged_df.head())


def selected():
    file1_path = 'D:/GBMFeatures/FEATURES_FINAL/MGMT/final_features_MGMT_60.csv'
    df1 = pd.read_csv(file1_path)
    file2_path = 'D:/flask/uploads/final_features.csv'
    df2 = pd.read_csv(file2_path)
    common_columns = list(set(df1.columns).intersection(df2.columns))
    selected_df = df1[common_columns]
    selected_df.to_csv('D:/flask/uploads/final_features_MGMT.csv', index=False)
    print(selected_df.head())


def predict():

    
    rename()
    concatenate()
    selected()
    class_weights = {0: 1, 1: 1.3}

    target = 'MGMT'
    n_features = 60

    concatenated_features = pd.read_csv(f"D:/flask/uploads/final_features_MGMT.csv")
    concatenated_features_mgmt = concatenated_features
    target_df = pd.read_csv("D:/GBMFeatures/MGMT_OS.csv")[target]

# Step 6: Train an SVM model using Stratified Kfold cross validation and grid search cv
    scaler = StandardScaler()
    svc = SVC(probability=True, class_weight=class_weights)

    param_grid = {
    'C': [0.1, 1, 10,100],
    'kernel': ['linear','poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    best_svc = None
    best_score = 0
    best_scaler = None

    # Initialize empty arrays to store the true and predicted labels
    all_y_test = np.array([])
    all_y_pred = np.array([])

    for i, (train, test) in enumerate(cv.split(concatenated_features, target_df)):
        X_train = concatenated_features.iloc[train,:]
        y_train = target_df.iloc[train].values.ravel()
        X_test = concatenated_features.iloc[test,:]
        y_test = target_df.iloc[test].values.ravel()
        grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='roc_auc',verbose=0)
    
    # Fit the scaler on the training data
        scaler.fit(X_train)

    # Transform the training and testing data
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        grid_search.fit(X_train_scaled, y_train)
        best_svc_fold = grid_search.best_estimator_
    
        score = grid_search.best_score_
    print(score)
    print(best_svc_fold.predict(X_test_scaled))
    arr = np.array(best_svc_fold.predict(X_test_scaled))
    predict=np.mean(arr)
    print(predict)
    return predict
