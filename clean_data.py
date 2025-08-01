import pandas as pd

def clean_data(input_path, output_path):
    """Remove problematic rows from dataset"""
    try:
        # Load data
        df = pd.read_csv(input_path)
        
        print(f"Original rows: {len(df)}")
        
        # 1. Remove rows with missing labels
        df = df.dropna(subset=['label'])
        
        # 2. Fill missing text with empty string
        df['text'] = df['text'].fillna('')
        
        # 3. Remove duplicate rows
        df = df.drop_duplicates()
        
        print(f"Clean rows: {len(df)}")
        print("\nCleaned data summary:")
        print(df.isna().sum())
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"\nSaved clean data to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    clean_data(
        input_path='data/spam_ham_dataset.csv',
        output_path='data/cleaned_spam_ham.csv'
    )