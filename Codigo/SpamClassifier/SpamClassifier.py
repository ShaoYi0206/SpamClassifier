import string
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

# Pré-Processamento de dados - Remover números, pontuações, tokenizar o texto, remover stopwords, PorterStemmer
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

# Separacao das colunas de um DF
def separation(df):
    X = df['Message']
    y = df['Category']
    return X, y

# Avaliando a performance dos métodos de ML 
def evaluate_performance(X, y, method, method_name):
    
    result_balanced_accuracies = []
    result_precisions = []
    result_recalls = []
    result_f1_scores = []

    for i in range(1, 31):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        
        # Listas temporárias
        balanced_accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        # Validação cruzada
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Vetorizacao TF-IDF
            vectorizer = TfidfVectorizer()
            X_train_transformed = vectorizer.fit_transform(X_train)  
            X_test_transformed = vectorizer.transform(X_test)        
            
            # Treinamento do classificador
            classifier = method()
            classifier.fit(X_train_transformed, y_train)
            y_pred = classifier.predict(X_test_transformed)

            # Previsão e avaliação
            balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, pos_label = 1))
            recalls.append(recall_score(y_test, y_pred, pos_label = 1))
            f1_scores.append(f1_score(y_test, y_pred, pos_label = 1))

        # Calcular as médias das métricas dos 5 folds obtidos
        result_balanced_accuracies.append(np.mean(balanced_accuracies))
        result_precisions.append(np.mean(precisions))
        result_recalls.append(np.mean(recalls))
        result_f1_scores.append(np.mean(f1_scores))        
        
    # Calcular as médias finais das métricas obtidas no intervalo de 30 valores do random_state    
    mean_balanced_accuracy = np.mean(result_balanced_accuracies)
    mean_precision = np.mean(result_precisions)
    mean_recall = np.mean(result_recalls)
    mean_f1_score = np.mean(result_f1_scores)
    
    return [method_name, mean_balanced_accuracy, mean_precision, mean_recall, mean_f1_score]

# Avaliando cada modelo chamando a função evaluate_performance()
def evaluate_methods(X, y, methods, dataset_name):
    
    results = []
    print(f"Iniciando a avaliacao do conjunto: {dataset_name}")
    
    for method, method_name in methods:
        print(f"Avaliando o método: {method_name}")
        result = evaluate_performance(X, y, method, method_name)
        results.append([dataset_name] + result)
    
    return results

# Salvando os resultados em um DF
def report(results, file_name = None):
    
    df_results = pd.DataFrame(results, columns=['Dataset', 'Method', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    if file_name:
        file_path = os.path.join('Resultados/Desempenho', file_name)
        df_results.to_csv(file_path, index = False)

# Gerando gráficos
def plt_graph(df_results, file_name = None):
    
    plt.figure(figsize = (11, 8))
    
    ax = sns.barplot(x = 'Method', y = 'F1 Score', data = df_results, palette = 'viridis')
    
    # Configurações do gráfico
    plt.title(f'F1 Score dos Métodos para {df_results['Dataset'][0]}')
    plt.xlabel('Método')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    fontsize=12, color='black', 
                    xytext=(0, 9), textcoords='offset points')
    
    if file_name:
        file_path = os.path.join('Resultados/Gráficos', f'{file_name}.png')
        plt.savefig(file_path)
    plt.show()    
        

if __name__ == "__main__":
    results_performance = []
    
    # Metodos com os hiperparâmetros para dataset Spam Email
    methods_1 = [
        (lambda: MultinomialNB(alpha = 0.1), "MultinomialNB"),
        (lambda: SVC(C=10, gamma = 'scale', kernel='linear', random_state=42), "SVC"),
        (lambda: RandomForestClassifier(n_estimators=150, random_state=42), "RandomForest"),
        (lambda: KNeighborsClassifier(leaf_size = 30, n_neighbors=3, weights='distance'), "KNeighbors"),
        (lambda: DecisionTreeClassifier(splitter='random'), "Decision Tree"),
        (lambda: MLPClassifier(max_iter=500, random_state=42, warm_start=True), "Artificial Neural Networks")
    ]

    '''
    Conjunto Spam Email - https://www.kaggle.com/datasets/mfaisalqureshi/spam-email
    https://www.researchgate.net/publication/367918824_Email_Spam_Detection_using_Machine_Learning_Techniques
    '''
    df_1 = pd.read_csv('SpamClassifier/Datasets/SpamEmail.csv')
    # Definindo ham = 0 e spam = 1
    df_1['Category'] = df_1['Category'].apply(lambda x: 0 if x == 'ham' else 1)
    df_1 = process_data(df_1)
    
    X_1, y_1 = separation(df_1)
    results_performance = evaluate_methods(X_1, y_1, methods_1, 'SpamEmail')
    
    report(results_performance, 'results_performance_SpamEmail.csv')
    
    
    
    # Metodos com os hiperparâmetros para dataset Spam_email_Dataset
    methods_2 = [
        (lambda: MultinomialNB(alpha = 0.1), "MultinomialNB"),
        (lambda: SVC(C = 10, gamma = 'scale', kernel='linear', random_state = 42), "SVC"),
        (lambda: RandomForestClassifier(max_depth= None, n_estimators = 100, random_state=42), "RandomForest"),
        (lambda: KNeighborsClassifier(leaf_size = 30, n_neighbors = 3, weights='distance'), "KNeighbors"),
        (lambda: DecisionTreeClassifier(criterion='entropy', max_depth = None, min_samples_leaf = 1, min_samples_split = 2, splitter='random'), "Decision Tree"),
        (lambda: MLPClassifier(activation = 'relu', alpha = 0.001, hidden_layer_sizes = (50,), learning_rate = 'constant', max_iter = 500, random_state = 42, solver = 'adam', warm_start = True), "Artificial Neural Networks")
    ]
    
    ''' 
    Conjunto Spam email Dataset - https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset?resource=download
    Artigo comparacao: https://www.sciencedirect.com/science/article/pii/S1877050921007493 
    Outro: https://digitalcommons.kennesaw.edu/cgi/viewcontent.cgi?article=1354&context=undergradsymposiumksu
    https://www.sciencedirect.com/science/article/pii/S1877050921013016
    '''
    df_2 = pd.read_csv('SpamClassifier/Datasets/Spam_email_Dataset.csv')
    # Nesse conjunto, ham = 0 e spam = 1
    df_2 = df_2.rename(columns={"text": "Message", "spam": "Category"})
    df_2 = process_data(df_2)
    
    X_2, y_2 = separation(df_2)
    results_performance = evaluate_methods(X_2, y_2, methods_2, 'Spam_email_Dataset')   
    
    report(results_performance, 'results_performance_Spam_email_Dataset.csv')
    
    
 
    # Metodos com os hiperparâmetros para dataset SMS_Spam
    methods_3 = [
        (lambda: MultinomialNB(alpha = 0.1), "MultinomialNB"),
        (lambda: SVC(C = 10, gamma = 'scale', kernel = 'linear', random_state = 42), "SVC"),
        (lambda: RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = 42), "RandomForest"),
        (lambda: KNeighborsClassifier(leaf_size = 30, n_neighbors = 10, weights = 'distance'), "KNeighbors"),
        (lambda: DecisionTreeClassifier(criterion = 'gini', max_depth = None, min_samples_leaf = 1, min_samples_split = 5, splitter = 'random'), "Decision Tree"),
        (lambda: MLPClassifier(activation = 'relu', alpha = 0.0001, hidden_layer_sizes = (100,), learning_rate = 'constant', max_iter = 500, random_state=42, solver = 'adam', warm_start = True), "Artificial Neural Networks")
    ]
    
    '''
    Conjunto SMS Spam - https://www.kaggle.com/datasets/shravan3273/sms-spam
    '''
    #spam = 1 e ham = 0
    df_3 = pd.read_csv('SpamClassifier\Datasets\SMS_Spam.csv')
    df_3 = df_3.rename(columns={"type":"Category", "text": "Message"})
    df_3 = process_data(df_3) 
    
    X_3, y_3 = separation(df_3)
    results_performance = evaluate_methods(X_3, y_3, methods_3, 'SMS_Spam')
    
    report(results_performance, 'results_performance_SMS_Spam.csv')
    
    
    # Gerar e salvar os gráficos
    plt_graph(pd.read_csv('Resultados/Desempenho/results_performance_SpamEmail.csv'), 'graph_SpamEmail')
    plt_graph(pd.read_csv('Resultados/Desempenho/results_performance_SMS_Spam.csv'), 'graph_SMS_Spam')
    plt_graph(pd.read_csv('Resultados/Desempenho/results_performance_Spam_email_Dataset.csv'), 'graph_Spam_email_Dataset')
    