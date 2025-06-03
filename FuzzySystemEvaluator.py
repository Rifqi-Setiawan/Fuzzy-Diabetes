class FuzzySystemEvaluator:
    def __init__(self, mamdani_system, sugeno_system):
        self.mamdani = mamdani_system
        self.sugeno = sugeno_system
        
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluasi kinerja kedua sistem fuzzy"""
        # Prediksi dengan Mamdani
        mamdani_preds = []
        for _, row in X_test.iterrows():
            inputs = row.to_dict()
            pred = self.mamdani.predict(inputs)
            mamdani_preds.append(1 if pred > threshold else 0)
            
        # Prediksi dengan Sugeno
        sugeno_preds = []
        for _, row in X_test.iterrows():
            inputs = row.to_dict()
            pred = self.sugeno.predict(inputs)
            sugeno_preds.append(1 if pred > threshold else 0)
            
        # Hitung metrik evaluasi
        y_true = y_test.values
        
        mamdani_metrics = {
            'accuracy': accuracy_score(y_true, mamdani_preds),
            'f1_score': f1_score(y_true, mamdani_preds)
        }
        
        sugeno_metrics = {
            'accuracy': accuracy_score(y_true, sugeno_preds),
            'f1_score': f1_score(y_true, sugeno_preds)
        }
        
        return {
            'Mamdani': mamdani_metrics,
            'Sugeno': sugeno_metrics
        }
        
    def compare_performance(self, metrics):
        """Membandingkan kinerja kedua metode"""
        print("\nPerbandingan Kinerja:")
        print("Metrik\t\tMamdani\t\tSugeno")
        print("------------------------------------")
        for metric in metrics['Mamdani'].keys():
            mamdani_val = metrics['Mamdani'][metric]
            sugeno_val = metrics['Sugeno'][metric]
            print(f"{metric}\t\t{mamdani_val:.4f}\t\t{sugeno_val:.4f}")
            
        # Analisis sederhana
        print("\nAnalisis:")
        if metrics['Mamdani']['accuracy'] > metrics['Sugeno']['accuracy']:
            print("- Mamdani lebih akurat dalam klasifikasi")
        else:
            print("- Sugeno lebih akurat dalam klasifikasi")
            
        if metrics['Mamdani']['f1_score'] > metrics['Sugeno']['f1_score']:
            print("- Mamdani memiliki keseimbangan precision-recall yang lebih baik")
        else:
            print("- Sugeno memiliki keseimbangan precision-recall yang lebih baik")

# Contoh penggunaan lengkap
if __name__ == "__main__":
    # 1. Prapemrosesan data
    preprocessor = DataPreprocessor('dataset.csv')
    data = preprocessor.load_data()
    data = preprocessor.handle_missing_values(strategy='mean')
    
    # Pilih fitur dan target (sesuaikan dengan dataset Anda)
    features = ['feature1', 'feature2', 'feature3']
    target = 'target'
    X, y = preprocessor.select_features(features, target)
    X = preprocessor.normalize_data()
    X_train, X_test, y_train, y_test = preprocessor.split_data()
    
    # 2. Membangun sistem Mamdani
    mamdani = MamdaniFuzzySystem()
    
    # Tambahkan variabel input dan fungsi keanggotaan
    mamdani.add_input_variable('feature1', 0, 1)
    mamdani.add_input_membership('feature1', 'low', 'triangular', [0, 0, 0.5])
    mamdani.add_input_membership('feature1', 'medium', 'triangular', [0, 0.5, 1])
    mamdani.add_input_membership('feature1', 'high', 'triangular', [0.5, 1, 1])
    
    mamdani.add_input_variable('feature2', 0, 1)
    mamdani.add_input_membership('feature2', 'low', 'triangular', [0, 0, 0.5])
    mamdani.add_input_membership('feature2', 'medium', 'triangular', [0, 0.5, 1])
    mamdani.add_input_membership('feature2', 'high', 'triangular', [0.5, 1, 1])
    
    # Tambahkan variabel output dan fungsi keanggotaan
    mamdani.add_output_variable('output', 0, 1)
    mamdani.add_output_membership('output', 'low', 'triangular', [0, 0, 0.5])
    mamdani.add_output_membership('output', 'medium', 'triangular', [0, 0.5, 1])
    mamdani.add_output_membership('output', 'high', 'triangular', [0.5, 1, 1])
    
    # Tambahkan rules (sesuaikan dengan domain masalah)
    mamdani.add_rule([('feature1', 'low'), ('feature2', 'low')], [('output', 'low')])
    mamdani.add_rule([('feature1', 'medium'), ('feature2', 'medium')], [('output', 'medium')])
    mamdani.add_rule([('feature1', 'high'), ('feature2', 'high')], [('output', 'high')])
    
    # 3. Membangun sistem Sugeno
    sugeno = SugenoFuzzySystem()
    
    # Tambahkan variabel input (sama seperti Mamdani)
    sugeno.add_input_variable('feature1', 0, 1)
    sugeno.add_input_membership('feature1', 'low', 'triangular', [0, 0, 0.5])
    sugeno.add_input_membership('feature1', 'medium', 'triangular', [0, 0.5, 1])
    sugeno.add_input_membership('feature1', 'high', 'triangular', [0.5, 1, 1])
    
    sugeno.add_input_variable('feature2', 0, 1)
    sugeno.add_input_membership('feature2', 'low', 'triangular', [0, 0, 0.5])
    sugeno.add_input_membership('feature2', 'medium', 'triangular', [0, 0.5, 1])
    sugeno.add_input_membership('feature2', 'high', 'triangular', [0.5, 1, 1])
    
    # Tambahkan variabel output (untuk Sugeno, bisa berupa nilai konstan atau fungsi)
    sugeno.add_output_variable('output', default_value=0)
    
    # Tambahkan rules dengan consequent berupa nilai konstan
    sugeno.add_rule([('feature1', 'low'), ('feature2', 'low')], [('output', 0.2)])
    sugeno.add_rule([('feature1', 'medium'), ('feature2', 'medium')], [('output', 0.5)])
    sugeno.add_rule([('feature1', 'high'), ('feature2', 'high')], [('output', 0.8)])
    
    # 4. Evaluasi kinerja
    evaluator = FuzzySystemEvaluator(mamdani, sugeno)
    metrics = evaluator.evaluate(X_test, y_test)
    evaluator.compare_performance(metrics)