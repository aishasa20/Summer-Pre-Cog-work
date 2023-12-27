
import numpy as np
import pandas as pd
import mne
import os
import pickle
import matplotlib.pyplot as plt

from scipy.signal import welch, spectrogram, stft

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Setting the random seed for reproducibility
np.random.seed(42)

class feature_extraction:
    def __init__(self, feature_type: str="mean") -> None:
        """
        Constructor for the feature_extraction class
        """
        self.feature_type = feature_type
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        if self.feature_type == "mean":
            return np.mean(data, axis=1)
        else:
            raise Exception("Invalid feature type")
        
class ClassicalML:
    def __init__(self, model_type: str="SVM") -> None:
        """
        Constructor for the ClassicalML class
        """
        self.model_type = model_type

        if self.model_type == "SVM":
            self.model = SVC()
        elif self.model_type == "LR":
            self.model = LogisticRegression()
        elif self.model_type == "RF":
            self.model = RandomForestClassifier()
        elif self.model_type == "KNN":
            self.model = KNeighborsClassifier()
        else:
            raise Exception("Invalid model type")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Function to fit the model
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Function to predict the labels
        """
        return self.model.predict(X_test)

    def get_metrics(self, y_test: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Function to get the metrics
        """
        metrics = {}

        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred)
        metrics["recall"] = recall_score(y_test, y_pred)
        metrics["f1"] = f1_score(y_test, y_pred)

        return metrics
        
def leave_out_one_validation(subjects: list=[], epoched_data_path: str="", save_path: str=""):
    """
    Function to perform leave-one-out cross validation
    """
    # Load the epoched data
    epoched_data = {}

    for subject in subjects:
        epoched_data_file = "{}/{}_epoch_data.pkl".format(epoched_data_path, subject)
        epoched_data[subject] = pd.read_pickle(epoched_data_file)
    
    metrics = {}

    # Compute PSD features for the epoched data
    # for keys in epoched_data.keys():
    #     epoched_data[keys]["congruent_psd"] = []
    #     epoched_data[keys]["incongruent_psd"] = []

    #     for epoch in epoched_data[keys]["congruent_epochs"]:
    #         f, Pxx = welch(epoch[:128, :], fs=512, nperseg=512)
    #         epoched_data[keys]["congruent_psd"].append(Pxx)
        
    #     for epoch in epoched_data[keys]["incongruent_epochs"]:
    #         f, Pxx = welch(epoch[:128, :], fs=512, nperseg=512)
    #         epoched_data[keys]["incongruent_psd"].append(Pxx)
        
    #     epoched_data[keys]["congruent_psd"] = np.array(epoched_data[keys]["congruent_psd"])
    #     epoched_data[keys]["incongruent_psd"] = np.array(epoched_data[keys]["incongruent_psd"])

    #     print(epoched_data[keys]["congruent_psd"].shape)
    #     print(epoched_data[keys]["incongruent_psd"].shape)

    # Loop through the subjects
    for subject in subjects:
        # Get the training data
        training_data = {}
        for training_subject in subjects:
            if training_subject != subject:
                training_data[training_subject] = epoched_data[training_subject]
        
        # Get the testing data
        testing_data = epoched_data[subject]

        # Train a SVM model
        # Get the training data
        X_train = []
        y_train = []

        for training_subject in training_data.keys():
            congruent_epochs = training_data[training_subject]["congruent_epochs"]
            incongruent_epochs = training_data[training_subject]["incongruent_epochs"]

            X_train.extend(congruent_epochs)
            X_train.extend(incongruent_epochs)

            y_train.extend([1] * len(congruent_epochs))
            y_train.extend([0] * len(incongruent_epochs))
        
        # Get the testing data
        X_test = []
        y_test = []

        congruent_epochs = testing_data["congruent_epochs"]
        incongruent_epochs = testing_data["incongruent_epochs"]

        X_test.extend(congruent_epochs)
        X_test.extend(incongruent_epochs)

        y_test.extend([1] * len(congruent_epochs))
        y_test.extend([0] * len(incongruent_epochs))

        # Convert the data to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Aggregate the data along channels
        X_train = np.mean(X_train[:,:128,:], axis=1)
        X_test = np.mean(X_test[:,:128,:], axis=1)

        print(X_train.shape)

        # Reshape the data
        X_train = X_train.reshape(X_train.shape[0], -1)

        X_test = X_test.reshape(X_test.shape[0], -1)

        # Create a pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("RF", SVC())
        ])

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Get the predictions
        y_pred = pipeline.predict(X_test)

        # Get the metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print the metrics
        print("Subject: {}".format(subject))
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))

        # Save the metrics
        metrics[subject] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Print average metrics
    print("Average metrics:")
    print("Accuracy: {}".format(np.mean([metrics[subject]["accuracy"] for subject in subjects])))
    print("Precision: {}".format(np.mean([metrics[subject]["precision"] for subject in subjects])))
    print("Recall: {}".format(np.mean([metrics[subject]["recall"] for subject in subjects])))
    print("F1: {}".format(np.mean([metrics[subject]["f1"] for subject in subjects])))

    # Save the metric file
    with open("{}/metrics.pkl".format(save_path), "wb") as f:
        pickle.dump(metrics, f)
    
    


if __name__ == "__main__":
    epoched_data_path = "/media/data/PRECOG_Data/2022N400_Epoched/"

    # Get the list of subjects
    # Exclude subject 5, 10, 15 and 18
    
    subjects = ["sub-{:02d}".format(i) for i in range(1, 25) if i not in [5, 10, 15, 18]]

    leave_out_one_validation(subjects=subjects, epoched_data_path=epoched_data_path)

    
