import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        self.feature_names = None
        
    def load_data(self, file_path, file_type='csv', **kwargs):
        """Memuat data dari berbagai format file"""
        try:
            if file_type.lower() == 'csv':
                data = pd.read_csv(file_path, **kwargs)
            elif file_type.lower() == 'excel':
                data = pd.read_excel(file_path, **kwargs)
            elif file_type.lower() == 'json':
                data = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Format file {file_type} tidak didukung")
                
            print(f"Data berhasil dimuat: {data.shape}")
            return data
        except Exception as e:
            print(f"Error memuat data: {e}")
            return None
            
    def explore_data(self, data):
        """Eksplorasi data dasar"""
        print("=== INFORMASI DATASET ===")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print("\n=== INFO DATA ===")
        print(data.info())
        
        print("\n=== STATISTIK DESKRIPTIF ===")
        print(data.describe())
        
        print("\n=== MISSING VALUES ===")
        missing = data.isnull().sum()
        print(missing[missing > 0])
        
        print("\n=== TIPE DATA ===")
        print(data.dtypes)
        
        return {
            'shape': data.shape,
            'columns': list(data.columns),
            'missing_values': missing,
            'dtypes': data.dtypes
        }
        
    def handle_missing_values(self, data, strategy='mean', columns=None):
        """Menangani missing values"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        if strategy in ['mean', 'median', 'most_frequent']:
            self.imputer = SimpleImputer(strategy=strategy)
            data[columns] = self.imputer.fit_transform(data[columns])
        elif strategy == 'drop':
            data = data.dropna(subset=columns)
        elif strategy == 'fill_zero':
            data[columns] = data[columns].fillna(0)
        else:
            raise ValueError(f"Strategy {strategy} tidak didukung")
            
        print(f"Missing values ditangani dengan strategy: {strategy}")
        return data
        
    def encode_categorical(self, data, columns=None, method='label'):
        """Encoding variabel kategorikal"""
        if columns is None:
            columns = data.select_dtypes(include=['object']).columns
            
        for col in columns:
            if method == 'label':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
            elif method == 'onehot':
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data, dummies], axis=1)
                data = data.drop(col, axis=1)
                
        print(f"Categorical encoding selesai dengan method: {method}")
        return data
        
    def scale_features(self, data, columns=None, method='standard'):
        """Scaling fitur numerik"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Scaling method {method} tidak didukung")
            
        data[columns] = self.scaler.fit_transform(data[columns])
        print(f"Feature scaling selesai dengan method: {method}")
        return data
        
    def remove_outliers(self, data, columns=None, method='iqr', threshold=1.5):
        """Menghapus outliers"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        initial_shape = data.shape
        
        if method == 'iqr':
            for col in columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                data = data[(data[col] >= lower) & (data[col] <= upper)]
                
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < threshold]
                
        print(f"Outliers dihapus: {initial_shape[0] - data.shape[0]} rows")
        return data
        
    def split_data(self, data, target_column, test_size=0.2, random_state=42):
        """Split data menjadi train dan test"""
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} tidak ditemukan")
            
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.feature_names = list(X.columns)
        
        print(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    def create_fuzzy_sets(self, data, column, n_sets=3, set_names=None):
        """Membuat fuzzy sets berdasarkan distribusi data"""
        if column not in data.columns:
            raise ValueError(f"Column {column} tidak ditemukan")
            
        col_data = data[column]
        min_val = col_data.min()
        max_val = col_data.max()
        
        if set_names is None:
            set_names = [f"set_{i+1}" for i in range(n_sets)]
            
        if len(set_names) != n_sets:
            raise ValueError("Jumlah set_names harus sama dengan n_sets")
            
        # Buat fuzzy sets dengan overlapping triangular membership
        sets = {}
        step = (max_val - min_val) / (n_sets - 1)
        
        for i, name in enumerate(set_names):
            if i == 0:  # Set pertama
                a = min_val
                b = min_val
                c = min_val + step
            elif i == n_sets - 1:  # Set terakhir
                a = max_val - step
                b = max_val
                c = max_val
            else:  # Set tengah
                a = min_val + (i - 1) * step
                b = min_val + i * step
                c = min_val + (i + 1) * step
                
            sets[name] = ('triangular', [a, b, c])
            
        return sets
        
    def plot_data_distribution(self, data, columns=None, figsize=(15, 10)):
        """Plot distribusi data"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
            
        n_cols = min(4, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
            
        for i, col in enumerate(columns):
            row = i // n_cols
            col_idx = i % n_cols
            
            if n_rows > 1:
                ax = axes[row, col_idx]
            else:
                ax = axes[col_idx]
                
            data[col].hist(bins=30, ax=ax, alpha=0.7)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
        # Hide empty subplots
        for i in range(len(columns), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            if n_rows > 1:
                axes[row, col_idx].axis('off')
            else:
                axes[col_idx].axis('off')
                
        plt.tight_layout()
        plt.show()
        
    def correlation_matrix(self, data, figsize=(10, 8)):
        """Plot correlation matrix"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        plt.figure(figsize=figsize)
        correlation = numeric_data.corr()
        
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return correlation
        
    def preprocess_pipeline(self, data, target_column, 
                           missing_strategy='mean',
                           encoding_method='label',
                           scaling_method='standard',
                           remove_outliers_flag=True,
                           outlier_method='iqr',
                           test_size=0.2):
        """Pipeline lengkap untuk preprocessing data"""
        print("=== MEMULAI PREPROCESSING PIPELINE ===")
        
        # 1. Eksplorasi data
        self.explore_data(data)
        
        # 2. Handle missing values
        data = self.handle_missing_values(data, strategy=missing_strategy)
        
        # 3. Encode categorical variables
        data = self.encode_categorical(data, method=encoding_method)
        
        # 4. Remove outliers
        if remove_outliers_flag:
            data = self.remove_outliers(data, method=outlier_method)
        
        # 5. Scale features (sebelum split untuk menghindari data leakage)
        # Tapi kita akan fit scaler hanya pada training data nanti
        
        # 6. Split data
        X_train, X_test, y_train, y_test = self.split_data(data, target_column, test_size)
        
        # 7. Scale features pada training data dan transform test data
        if scaling_method:
            X_train_scaled = self.scale_features(X_train.copy(), method=scaling_method)
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns, index=X_test.index)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        print("=== PREPROCESSING PIPELINE SELESAI ===")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'processed_data': data,
            'feature_names': self.feature_names
        }
    
    def save_preprocessor(self, filepath):
        """Simpan preprocessor untuk digunakan nanti"""
        import pickle
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"Preprocessor disimpan ke {filepath}")
    
    def load_preprocessor(self, filepath):
        """Muat preprocessor yang sudah disimpan"""
        import pickle
        
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.imputer = preprocessor_data['imputer']
        self.feature_names = preprocessor_data['feature_names']
        
        print(f"Preprocessor dimuat dari {filepath}")
    
    def transform_new_data(self, data):
        """Transform data baru menggunakan preprocessor yang sudah di-fit"""
        # Apply imputer jika ada
        if self.imputer:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = self.imputer.transform(data[numeric_cols])
        
        # Apply label encoders
        for col, encoder in self.label_encoders.items():
            if col in data.columns:
                data[col] = encoder.transform(data[col].astype(str))
        
        # Apply scaler
        if self.scaler:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        
        return data

# Contoh penggunaan
if __name__ == "__main__":
    df = pd.read_csv('diabetes.csv')
    
    #ambil fitur
    selected_features = ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Outcome']
    df = df[selected_features]

    # Mengganti nilai 0 menjadi NaN pada fitur selain outcome
    for col in ['Glucose', 'BloodPressure', 'BMI']:
        df[col] = df[col].replace(0, np.nan)

    # Inisialisasi preprocessor
