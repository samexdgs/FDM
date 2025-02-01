import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
from src import config
from src.paths import MODEL_DIR

import pickle

final_model_result = []

def train_model(model, Xtrain, ytrain, Xtest, ytest, model_name, save_ckpt=True):
    
    categorical_preprocessor_1 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), 
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    
    categorical_prepocessor_2 = Pipeline(
        steps=[
            ("vectorize", TfidfVectorizer(ngram_range=(1, 2)))
        ]
    )
    
    data_preprocessor = ColumnTransformer(
        transformers=[
            ("cat_pre_1", categorical_preprocessor_1, config.CAT_COLS), 
            ("cat_pre_2", categorical_prepocessor_2, config.TEXT_COLS[0])
        ],
        remainder="passthrough"
    )
    
    pipeline = Pipeline(
        steps=[
            ("data_preprocessing", data_preprocessor),
            ("model", model)
        ]
    )
    
    pipeline.fit(Xtrain, ytrain)
    
    y_pred = pipeline.predict(Xtest)
    y_pred_proba = pipeline.predict_proba(Xtest)
        
    
    accuracy = accuracy_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred, average="macro")
    precision = precision_score(ytest, y_pred, average="macro")
    f1 = f1_score(ytest, y_pred, average="macro")
    roc_auc = roc_auc_score(ytest, y_pred_proba, multi_class='ovr', average='macro')
    report = classification_report(ytest, y_pred)

    print(f"Accuracy: {accuracy:.2f} | Recall: {recall:.2f} | Precsion: {precision:.2f} | ROC_AUC: {roc_auc:.2f} | F1-score: {f1:.2f}")
    print("_______" * 12, '\n')
    print(report)
    print("_______" * 12, '\n')
    cm = confusion_matrix(ytest, y_pred, normalize="true")
    sns.heatmap(cm, annot=True, cbar=False, cmap='coolwarm')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    model_result = {
        "Model": model_name,
        "Accuracy": round(accuracy, 2), 
        "Recall": round(recall, 2), 
        "Precision":round(precision, 2),
        "F1-Score": round(f1, 2),
        "ROC-AUC" : round(roc_auc, 2)
        }
  
    if not final_model_result:
        final_model_result.append(model_result)
    else:
        for i, result in enumerate(final_model_result):
            if result.get("Model") == model_name:
                final_model_result[i] = model_result
                break
        else:
            final_model_result.append(model_result)
            
    if save_ckpt:
        with open(f"{MODEL_DIR}/{model_name.lower().replace(' ', '_')}.pkl", "wb") as file:
            pickle.dump(pipeline, file)

    return pipeline, final_model_result