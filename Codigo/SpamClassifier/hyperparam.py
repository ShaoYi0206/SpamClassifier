import string
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

def process_data(df):
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    df['Message'] = df['Message'].str.lower()
    df['Message'] = df['Message'].apply(lambda x: re.sub(r'\d+', '', x))
    df['Message'] = df['Message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['Message'] = df['Message'].apply(word_tokenize)
    df['Message'] = df['Message'].apply(lambda x: [word for word in x if word not in stop_words])
    df['Message'] = df['Message'].apply(lambda x: [ps.stem(word) for word in x])
    df['Message'] = df['Message'].apply(lambda x: " ".join(x))
    return df

# Separar preditores e alvo
def separation(df):
    X = df['Message']
    y = df['Category']
    return X, y

def grid_search(X_transformed, y_train, method, method_name, param_grid, skf, dataset_name):
    
    result = []
    print(f"Otimizando hiperparâmetros para: {method_name}")
    
    grid_search = GridSearchCV(
        estimator   =   method(),
        param_grid  =   param_grid[method_name],
        scoring     =   'f1',
        cv          =   skf,
        n_jobs      =   -1,
        verbose     =   2,
    )
    
    grid_search.fit(X_transformed, y_train)
    
    best_params     = grid_search.best_params_
    best_score      = grid_search.best_score_
    best_estimator  = grid_search.best_estimator_
     
    result.append((dataset_name, method_name, best_score, best_params, best_estimator))
        
    return result

def evaluate_methods(X, y, methods, param_grid, dataset_name):
    print(f"Iniciando a avaliação do conjunto: {dataset_name}")
    
    results = []
    for method, method_name in methods:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            vectorizer = TfidfVectorizer()
            X_train_transformed = vectorizer.fit_transform(X_train)

            results += grid_search(X_train_transformed, y_train, method, method_name, param_grid, skf, dataset_name)    
    
    return results

# Relatar os resultados
def report(results, filename=None):
    df_results = pd.DataFrame(results, columns=[
        'Dataset', 'Method', 'Best Score', 'Best Parameter', 'Best Estimator'
    ])
    if filename:
        df_results.to_csv(filename, index=False)


methods = [
    #(MultinomialNB, "MultinomialNB"),
    #(SVC, "SVC"),
    #(RandomForestClassifier, "RandomForestClassifier"),
    #(KNeighborsClassifier, "KNeighborsClassifier"),
    #(DecisionTreeClassifier, "DecisionTreeClassifier")
    (MLPClassifier, "MLPClassifier")
]

param_grid = {
    #"MultinomialNB": {"alpha": [0.1, 0.5, 1.0, 1.5, 2.0]},
    #"SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"], "random_state": [42]},
    #"RandomForestClassifier": {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20], "random_state": [42]},
    #"KNeighborsClassifier": {"n_neighbors": [3, 5, 10], "weights": [None, "uniform", "distance"], "leaf_size": [30, 40, 50]},
    #"DecisionTreeClassifier": {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best", "random"], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
    "MLPClassifier": {"activation": ["relu", "tanh"], "alpha": [0.0001, 0.001], "hidden_layer_sizes": [(50,), (100,)], "learning_rate": ["constant", "invscaling"], "max_iter": [500], "solver": ["sgd", "adam"], "random_state": [42], "warm_start": [True]}

}


if __name__ == "__main__":
    results = []
    
    #df_1 = pd.read_csv("SpamClassifier/Datasets/SpamEmail.csv")
    #df_1['Category'] = df_1['Category'].apply(lambda x: 0 if x == 'ham' else 1)
    #df = process_data(df_1)
    #X_1, y_1 = separation(df_1)
    #results += evaluate_methods(X_1, y_1, methods, param_grid, "SpamEmail.csv")
      
    df_2 = pd.read_csv('SpamClassifier/Datasets/Spam_email_Dataset.csv')
    df_2 = df_2.rename(columns={"text": "Message", "spam": "Category"})
    df_2 = process_data(df_2)
    X_2, y_2 = separation(df_2)
    results += evaluate_methods(X_2, y_2, methods, param_grid, 'Spam_email_Dataset.csv') 
    #
    df_3 = pd.read_csv('SpamClassifier\Datasets\SMS_Spam.csv')
    df_3 = df_3.rename(columns={"type":"Category", "text": "Message"})
    df_3 = process_data(df_3) 
    X_3, y_3 = separation(df_3) 
    results += evaluate_methods(X_3, y_3, methods, param_grid, 'SMS_Spam.csv')

    report(results, "results_hyperparam.csv")
