import pandas as pd

def inspect_data(data_path):
    """Analyze dataset for issues"""
    try:
        df = pd.read_csv(data_path)
        
        print("\n=== Data Quality Report ===")
        print(f"Total rows: {len(df)}")
        
        print("\nMissing values:")
        print(df.isna().sum())
        
        print("\nLabel distribution:")
        print(df['label'].value_counts(dropna=False))
        
        if df['text'].isna().any():
            print("\nSample rows with missing text:")
            print(df[df['text'].isna()].head())
            
        if df['label'].isna().any():
            print("\nSample rows with missing labels:")
            print(df[df['label'].isna()].head())
            
        print("\nSample valid data:")
        print(df.head(3))
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Make sure:")
        print("- The file exists at the specified path")
        print("- It has 'text' and 'label' columns")
        print("- The file is a valid CSV")

if __name__ == "__main__":
    inspect_data('data/spam_ham_dataset.csv')