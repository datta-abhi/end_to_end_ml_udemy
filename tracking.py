import numpy as np
import pandas as pd
import argparse
import itertools

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple,List

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

# function to pass arguments from command line
def parse_args():
    parser = argparse.ArgumentParser()
    
    # adding arguments as strings
    parser.add_argument("--penalty",  choices = ['l1','l2','elasticnet','none'],default = 'l2')
    parser.add_argument("-C", type = float, default=1.0)
    parser.add_argument("--solver", choices= ['newton-cg','sag','saga','liblinear','lbfgs'], default='lbfgs')
    
    return parser.parse_args()

# function to split data into train/ test
def prepare_data(df: pd.DataFrame,test_size: float)-> Tuple[pd.DataFrame,pd.DataFrame]:
    
    train_df, test_df = train_test_split(df,test_size=test_size, stratify= df['sentiment'],random_state=123)
    return train_df, test_df

# function for feature engg --> vectorizing reviews
def make_features(train_df: pd.DataFrame, test_df: pd.DataFrame)-> Tuple[csr_matrix,csr_matrix]:
    
    vectorizer = TfidfVectorizer(stop_words='english')
    train_inputs = vectorizer.fit_transform(train_df['review'])
    test_inputs = vectorizer.transform(test_df['review'])
    
    return train_inputs, test_inputs

# function to train the model
def train(train_inputs, train_outputs: np.ndarray, **model_kwargs)-> BaseEstimator:
    
    model = LogisticRegression(**model_kwargs)
    model.fit(train_inputs,train_outputs)
    return model

# function to draw confusion matrix
def draw_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray, class_names: List[str])-> Figure:
    
    labels = list(range(len(class_names)))
    conf_mat = confusion_matrix(true_labels,predicted_labels,labels = labels)
    
    plt.imshow(conf_mat, interpolation='nearest', cmap= plt.cm. Purples)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=20)
    plt.yticks(tick_marks, class_names, fontsize=20)

    fmt = 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black", fontsize=20)

    plt.title("Confusion matrix")
    plt.ylabel('Actual label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout()
    return plt.gcf()

# function to evaluate model
def evaluate(model: BaseEstimator, test_inputs: csr_matrix, test_outputs: np.ndarray, class_names: List[str])-> Tuple[float,Figure]:
    
    predicted_test_outputs = model.predict(test_inputs)
    figure = draw_confusion_matrix(test_outputs, predicted_test_outputs, class_names)
    f1 = f1_score(test_outputs,predicted_test_outputs)
    
    return f1, figure    

# main function
def main(args):
    df = pd.read_csv('./data/imdb_raw_data.csv')
    
    # mapping label
    label_map = {'negative':0, 'positive':1}
    df['label'] = df['sentiment'].map(label_map)
    
    test_size = 0.3
    train_df, test_df = prepare_data(df,test_size = test_size)
    
    # tracking experiments
    mlflow.set_experiment('tracking_demo')
    # set context manager
    with mlflow.start_run():
        
        # converting sentences to vectors using Tf-idf
        train_inputs, test_inputs = make_features(train_df,test_df)
        model = train(train_inputs, train_df['label'].values,
                    penalty = args.penalty, C = args.C, solver = args.solver)
        
        # model evaluation
        f1_score, figure = evaluate(model, test_inputs, test_df['label'].values, df['sentiment'].unique().tolist())
        figure.savefig("./confusion_matrix.png")
        print("f1 score: ",round(f1_score,3))
        
        mlflow.log_param('test_size',test_size)
        mlflow.log_metric('f1_score', f1_score)
        mlflow.log_param('C',args.C)
        mlflow.log_figure(figure,'figure.png')
    
if __name__=="__main__":
    main(parse_args())    