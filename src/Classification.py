from Gabor_Outputs_generation import read_from_csv,csv_name
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# In this file, we use the previously saved features and create the train-test split and apply the SVM
def main():

    classes_list,classes_names,_,usage, = read_from_csv(csv_name)

    phase = usage.values # phase means either training or testing phase
    labels = classes_list.values
    print(labels)
    
    if csv_name == "src\\data\\fer2013.csv":
        imgwise_blockmeans = np.load("src\\data\\imgwise_blockmeans_fer.npy")
    elif csv_name == "src\\data\\ckextended.csv":
        imgwise_blockmeans = np.load("src\\data\\imgwise_blockmeans_ck.npy")
    # Break the dataset in train and test
    X_train = []
    X_test = []
    labels_train = []
    labels_test = []
    for i,value in enumerate(phase):
        if value == 'Training':
            X_train.append(imgwise_blockmeans[i])
            labels_train.append(labels[i])
        else:
            X_test.append(imgwise_blockmeans[i])
            labels_test.append(labels[i])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    print(f"Shape of the training data = {X_train.shape}")
    print(f"Shape of testing data = {X_test.shape}")
    print(f"Label's shape = {labels.shape} and type of labels is {type(labels)}")

    # So each input image has 256 features. We will normalise each of the features

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit means it calculates the mean and variance and use them to normalise the features
    # Hence we "fit" on the training data and not on testing data. On the testing data we simply normalise using those mean and variance.
    X_test  = scaler.transform(X_test)

    ######################################### The Actual Classification ######################################### 

    #svm = SVC(kernel='rbf', C=10 , gamma='scale')
    svm = SVC(kernel='linear',C=1)
    svm.fit(X_train, labels_train)

    accuracy = svm.score(X_test, labels_test)
    print("Accuracy:", accuracy)

    y_pred = svm.predict(X_test)
    cm = confusion_matrix(labels_test, y_pred)
    print(f"Confusion Matrix :\n {cm}")


    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes_names,
            yticklabels=classes_names)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    cr = classification_report(labels_test, y_pred,output_dict=True)
    print(f"Classification Report :\n {cr}")
    cr_df = pd.DataFrame(cr).transpose()

    plt.figure(figsize=(8,6))
    sns.heatmap(cr_df.iloc[:-1, :-1], annot=True, cmap='Blues')

    plt.title('Classification Report')
    plt.show()
        

if __name__ == "__main__":
    main()