import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def run_preprocessing(input_url, output_dir):
    print("Memulai proses otomatisasi preprocessing...")
    
    # 1. Memuat Dataset
    df = pd.read_csv(input_url)
    df_clean = df.copy()

    # 2. Drop kolom customerID jika ada
    if 'customerID' in df_clean.columns:
        df_clean.drop(columns=['customerID'], inplace=True)

    # 3. Menangani nilai kosong pada TotalCharges
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].astype(float)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())

    # 4. Encoding Data Kategorikal
    le = LabelEncoder()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = le.fit_transform(df_clean[col])

    # 5. Memisahkan Fitur dan Target
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']

    # 6. Splitting Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. Standardisasi Nilai Fitur Numerik
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 8. Menyimpan Output Hasil Preprocessing
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    print(f"Otomatisasi selesai! File disimpan di folder: '{output_dir}'")

if __name__ == "__main__":
    DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    OUTPUT_FOLDER = "namadataset_preprocessing"
    run_preprocessing(DATA_URL, OUTPUT_FOLDER)