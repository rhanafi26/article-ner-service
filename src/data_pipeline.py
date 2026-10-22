import glob
import json
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
RANDOM_STATE = 42

def clean_text_quotes(text):
    if pd.isna(text):
        return text

    text = str(text).strip()
    text = re.sub(r"^['’]+", "", text)
    return text.strip()

def load_and_profile_data():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    dataframes = []
    for file in csv_files:
        df_temp = pd.read_csv(file, sep=';', quotechar='"', on_bad_lines='skip')
        dataframes.append(df_temp)

    df = pd.concat(dataframes, ignore_index=True)

    df['title'] = df['title'].apply(clean_text_quotes)
    df['body'] = df['body'].apply(clean_text_quotes)

    total_rows = len(df)
    print(f"Total rows in the dataset: {total_rows}")

    duplicates = df.duplicated(subset=['title', 'body']).sum()
    print(f"Number of duplicate rows based on 'title' and 'body': {duplicates}")
    df_clean = df.drop_duplicates(subset=['title', 'body']).copy()

    print(f"Missing values in 'title': {df_clean['title'].isna().sum()}")
    print(f"Missing values in 'body': {df_clean['body'].isna().sum()}")
    df_clean = df_clean.dropna(subset=['title', 'body'])

    df_clean = df_clean[(df_clean['title'] != "") & (df_clean['body'] != "")]

    if 'language' in df_clean.columns:
        print("Language distribution:")
        print(df_clean['language'].value_counts())

    if 'entities' in df_clean.columns:
        print("Entities distribution:")
        print(df_clean['entities'].dropna().head(3))

    return df_clean

def split_and_save_data(df):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_STATE, shuffle=True
    )
    
    print(f"Train set: {len(train_df)} artikel (70%)")
    print(f"Validation set: {len(val_df)} artikel (15%)")
    print(f"Test set: {len(test_df)} artikel (15%)")
    
    columns_to_export = ['original_id', 'title', 'body', 'language']
    
    train_df[columns_to_export].to_json(
        os.path.join(OUTPUT_DIR, "train_split.json"), orient="records", indent=2, force_ascii=False
    )
    val_df[columns_to_export].to_json(
        os.path.join(OUTPUT_DIR, "val_split.json"), orient="records", indent=2, force_ascii=False
    )
    test_df[columns_to_export].to_json(
        os.path.join(OUTPUT_DIR, "test_split.json"), orient="records", indent=2, force_ascii=False
    )
    
    eval_subset = test_df.head(10)[columns_to_export].to_dict(orient="records")
    eval_template = []
    for item in eval_subset:
        eval_template.append({
            "original_id": item['original_id'],
            "title": item['title'],
            "body": item['body'],
            "language": item['language'],
            "gold_entities": []
        })
        
    with open(os.path.join(OUTPUT_DIR, "evaluation_examples.json"), "w", encoding="utf-8") as f:
        json.dump(eval_template, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    df_processed = load_and_profile_data()
    split_and_save_data(df_processed)