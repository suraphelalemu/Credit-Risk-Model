import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class FeatureEngineering:
    """
    A class to perform feature engineering tasks for a given DataFrame.
    """

    @staticmethod
    def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates aggregate features such as total, average, count, and standard deviation of transaction amounts.
        """
        required_cols = ['CustomerId', 'TransactionId', 'Amount']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        agg_features = df.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),
            Average_Transaction_Amount=('Amount', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Transaction_Amount=('Amount', 'std')
        ).reset_index()

        df = df.merge(agg_features, on='CustomerId', how='left')
        return df

    @staticmethod
    def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts time-related features from the TransactionStartTime column.
        """
        if 'TransactionStartTime' not in df.columns:
            raise ValueError("Missing required column: TransactionStartTime")

        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
        df['Transaction_Day'] = df['TransactionStartTime'].dt.day
        df['Transaction_Month'] = df['TransactionStartTime'].dt.month
        df['Transaction_Year'] = df['TransactionStartTime'].dt.year

        return df

    @staticmethod
    def encode_categorical_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """
        Encodes categorical variables into numerical format using Label Encoding.
        """
        df = df.copy()
        label_encoder = LabelEncoder()

        for col in categorical_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            df[col] = label_encoder.fit_transform(df[col].astype(str))

        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handles missing values using a specified imputation strategy or drops them.
        """
        df = df.copy()
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        elif strategy == 'remove':
            df.dropna(inplace=True)
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'most_frequent', or 'remove'.")
        return df

    @staticmethod
    def normalize_numerical_features(df: pd.DataFrame, numerical_cols: list, method: str = 'standardize') -> pd.DataFrame:
        """
        Normalizes or standardizes numerical features.
        """
        df = df.copy()
        if method == 'standardize':
            scaler = StandardScaler()
        elif method == 'normalize':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be either 'standardize' or 'normalize'")

        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df


# ====== Example Usage ======
if __name__ == '__main__':
    print("🟢 Starting feature engineering process...")
    print("===============================================")

    # Load your dataset
    try:
        df = pd.read_csv('your_dataset.csv')  # Replace with your actual file path
    except FileNotFoundError:
        print("❌ Error: File not found.")
        exit()

    df_copy = df.copy().reset_index(drop=True)

    # Feature Engineering
    try:
        df_copy = FeatureEngineering.create_aggregate_features(df_copy)
        print("✅ Aggregate features created.")

        df_copy = FeatureEngineering.extract_time_features(df_copy)
        print("✅ Time features extracted.")

        df_copy = FeatureEngineering.encode_categorical_features(df_copy, ['Category'])  # replace 'Category' with your actual columns
        print("✅ Categorical features encoded.")

        df_copy = FeatureEngineering.handle_missing_values(df_copy, strategy='mean')
        print("✅ Missing values handled.")

        df_copy = FeatureEngineering.normalize_numerical_features(df_copy, ['Amount'])  # replace with your numerical columns
        print("✅ Numerical features normalized.")

    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        exit()

    print("✅ Feature engineering completed successfully.")
    print("===============================================")
