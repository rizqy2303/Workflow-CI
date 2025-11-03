import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import mlflow
import mlflow.sklearn
import warnings
import dagshub 
import time 
import os # Import os untuk path yang lebih aman

warnings.filterwarnings("ignore")

def preprocess_data(df):
    """Melakukan preprocessing sederhana: drop kolom, one-hot encode, dan fillna 0."""
    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df = df.fillna(0)
    return df

def main():
    
    # 1. INIT REPO (Menggunakan Variabel Lingkungan)
    # DAGSHUB_TOKEN harus diset sebagai GitHub Secret
    dagshub.init(repo_owner='rizqyputrap23', repo_name='my-first-repo', mlflow=True) 
    
    mlflow.set_experiment("Eksperimen_Advance_DagsHub_Final")
    
    # --- Data Loading dan Pemisahan ---
    # Asumsi: train.csv dan test.csv berada di root repositori
    file_train = "train.csv" 
    file_test = "test.csv" 
    TARGET_COLUMN = "Survived" 

    try:
        train_df = pd.read_csv(file_train) 
        test_df = pd.read_csv(file_test)
    except FileNotFoundError:
        print("ERROR: Pastikan file 'train.csv' dan 'test.csv' berada di folder yang sama dengan folder MLProject/.")
        return # Keluar dari fungsi jika file tidak ditemukan

    # Preprocessing
    combined_df = pd.concat([train_df.drop(columns=[TARGET_COLUMN], errors='ignore'), test_df], ignore_index=True)
    combined_processed = preprocess_data(combined_df)

    X_train_processed = combined_processed.iloc[:len(train_df)].drop(columns=[TARGET_COLUMN], errors='ignore')
    input_example = X_train_processed.head(1) 
    X_test_processed = combined_processed.iloc[len(train_df):].drop(columns=[TARGET_COLUMN], errors='ignore')
    y_train = train_df[TARGET_COLUMN]

    # Model Training (Grid Search)
    param_grid = {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
    logreg = LogisticRegression(random_state=42, solver='liblinear') 
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', verbose=0)
    
    print("Fitting Grid Search (30 fits)...")
    grid_search.fit(X_train_processed, y_train)

    best_model = grid_search.best_estimator_
    
    # Logic Evaluasi Data (Dipilih sesuai ketersediaan kolom target)
    if TARGET_COLUMN not in test_df.columns:
        X_eval = X_train_processed
        y_eval = y_train
        run_name_suffix = "Evaluasi_pada_TRAIN"
    else:
        # Peringatan: Anda mungkin perlu preprocessing test_df secara terpisah jika ingin evaluasi murni
        # Namun, kita ikuti logic penggabungan Anda untuk memastikan dimensi cocok
        X_eval = X_test_processed
        y_eval = test_df[TARGET_COLUMN]
        run_name_suffix = "Evaluasi_pada_TEST_Valid"
    
    
    # LOGGING KE DAGSHUB (Level Advance)
    # PENTING: Semua logging termasuk log_model harus di dalam blok 'with mlflow.start_run()'
    with mlflow.start_run(run_name=f"Best_LogReg_Advance_{run_name_suffix}"):
        
        y_pred = best_model.predict(X_eval)
        y_proba = best_model.predict_proba(X_eval)[:, 1] 
        
        # Hitung Metrik
        accuracy = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_eval, y_pred, average='binary', zero_division=0) 
        auc_score = roc_auc_score(y_eval, y_proba) 

        # 1. Logging PARAMETER
        mlflow.log_param("C", best_model.get_params()['C'])
        mlflow.log_param("penalty", best_model.get_params()['penalty'])

        # 2. Logging METRIK
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall_score", recall) 
        mlflow.log_metric("auc_roc_score", auc_score) 
        
        # 3. LOGGING ARTEFAK MODEL (KRUSIAL UNTUK ADVANCE/DOCKER)
        mlflow.sklearn.log_model(
            sk_model=best_model,  
            artifact_path="model", # Folder ini yang dicari oleh mlflow build-docker
            input_example=input_example # Logging Input Schema
        )
        
        print(f"\n--- Logging ke DagsHub Berhasil ---")
        print(f"Hyperparameter Terbaik: {grid_search.best_params_}")
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print("Model berhasil dicatat dalam format MLFlow standar untuk Dockerisasi.")


if __name__ == "__main__":
    main()
