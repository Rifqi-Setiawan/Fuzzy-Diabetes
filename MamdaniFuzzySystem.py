import numpy as np
import matplotlib.pyplot as plt

class MamdaniFuzzySystem:
    def __init__(self):
        self.rules = []
        self.input_variables = {}
        self.output_variables = {}
        
    def add_input_variable(self, name, min_val, max_val):
        """Menambahkan variabel input"""
        self.input_variables[name] = {'min': min_val, 'max': max_val, 'sets': {}}
        
    def add_output_variable(self, name, min_val, max_val):
        """Menambahkan variabel output"""
        self.output_variables[name] = {'min': min_val, 'max': max_val, 'sets': {}}
        
    def add_input_membership(self, var_name, set_name, mf_type, params):
        """Menambahkan fungsi keanggotaan untuk variabel input"""
        if var_name not in self.input_variables:
            raise ValueError(f"Variabel input {var_name} tidak ditemukan")
            
        self.input_variables[var_name]['sets'][set_name] = {
            'type': mf_type,
            'params': params
        }
        
    def add_output_membership(self, var_name, set_name, mf_type, params):
        """Menambahkan fungsi keanggotaan untuk variabel output"""
        if var_name not in self.output_variables:
            raise ValueError(f"Variabel output {var_name} tidak ditemukan")
            
        self.output_variables[var_name]['sets'][set_name] = {
            'type': mf_type,
            'params': params
        }
        
    def add_rule(self, antecedents, consequent):
        """Menambahkan rule fuzzy
        antecedents: list of tuples [(var_name, set_name), ...]
        consequent: tuple (var_name, set_name)
        """
        self.rules.append({'if': antecedents, 'then': consequent})
        
    def _membership_function(self, x, mf_type, params):
        """Menghitung nilai keanggotaan"""
        if isinstance(x, (list, np.ndarray)):
            return np.array([self._membership_function(xi, mf_type, params) for xi in x])
            
        if mf_type == 'triangular':
            a, b, c = params
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b < x < c:
                return (c - x) / (c - b)
                
        elif mf_type == 'trapezoidal':
            a, b, c, d = params
            if x <= a or x >= d:
                return 0.0
            elif a < x < b:
                return (x - a) / (b - a)
            elif b <= x <= c:
                return 1.0
            elif c < x < d:
                return (d - x) / (d - c)
                
        elif mf_type == 'gaussian':
            mean, sigma = params
            return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
            
        else:
            raise ValueError(f"Tipe fungsi keanggotaan {mf_type} tidak didukung")
            
    def fuzzify(self, inputs):
        """Fuzzifikasi input"""
        fuzzified = {}
        for var_name, value in inputs.items():
            if var_name not in self.input_variables:
                raise ValueError(f"Variabel input {var_name} tidak ditemukan")
                
            fuzzified[var_name] = {}
            for set_name, mf in self.input_variables[var_name]['sets'].items():
                membership = self._membership_function(value, mf['type'], mf['params'])
                fuzzified[var_name][set_name] = membership
                
        return fuzzified
    
    def infer(self, fuzzified_inputs):
        """Inferensi menggunakan metode Mamdani"""
        activated_rules = []
        
        for rule in self.rules:
            # Hitung firing strength menggunakan operator AND (minimum)
            strengths = []
            for var_name, set_name in rule['if']:
                if var_name in fuzzified_inputs and set_name in fuzzified_inputs[var_name]:
                    strength = fuzzified_inputs[var_name][set_name]
                    strengths.append(strength)
                else:
                    strengths.append(0.0)
                    
            rule_strength = min(strengths) if strengths else 0.0
            
            if rule_strength > 0:
                activated_rules.append({
                    'strength': rule_strength,
                    'consequent': rule['then']
                })
                
        return activated_rules
        
    def aggregate(self, activated_rules, output_var):
        """Agregasi menggunakan operator OR (maximum)"""
        if output_var not in self.output_variables:
            raise ValueError(f"Variabel output {output_var} tidak ditemukan")
            
        # Buat domain untuk output
        min_val = self.output_variables[output_var]['min']
        max_val = self.output_variables[output_var]['max']
        x = np.linspace(min_val, max_val, 1000)
        
        # Inisialisasi agregasi dengan zeros
        aggregated = np.zeros_like(x)
        
        for rule in activated_rules:
            strength = rule['strength']
            var_name, set_name = rule['consequent']
            
            if var_name == output_var and set_name in self.output_variables[var_name]['sets']:
                mf = self.output_variables[var_name]['sets'][set_name]
                membership_values = self._membership_function(x, mf['type'], mf['params'])
                
                # Clip membership dengan rule strength (implication)
                clipped = np.minimum(membership_values, strength)
                
                # Agregasi menggunakan maximum
                aggregated = np.maximum(aggregated, clipped)
                
        return x, aggregated
        
    def defuzzify(self, x, aggregated, method='centroid'):
        """Defuzzifikasi"""
        if method == 'centroid':
            if np.sum(aggregated) == 0:
                return np.mean(x)  # Return center of universe jika tidak ada aktivasi
            return np.sum(x * aggregated) / np.sum(aggregated)
            
        elif method == 'mean_of_max':
            max_val = np.max(aggregated)
            if max_val == 0:
                return np.mean(x)
            max_indices = np.where(aggregated == max_val)[0]
            return np.mean(x[max_indices])
            
        else:
            raise ValueError(f"Metode defuzzifikasi {method} tidak didukung")
        
    def predict(self, inputs, output_var=None, defuzz_method='centroid'):
        """Prediksi output untuk input tertentu"""
        try:
            if output_var is None:
                output_var = list(self.output_variables.keys())[0]
                
            fuzzified = self.fuzzify(inputs)
            activated_rules = self.infer(fuzzified)
            x, aggregated = self.aggregate(activated_rules, output_var)
            result = self.defuzzify(x, aggregated, defuzz_method)
            
            return result
        except Exception as e:
            print(f"Error dalam prediksi: {e}")
            return 0
            
    def plot_membership_functions(self, var_name, var_type='input'):
        """Plot fungsi keanggotaan"""
        if var_type == 'input' and var_name in self.input_variables:
            var_info = self.input_variables[var_name]
        elif var_type == 'output' and var_name in self.output_variables:
            var_info = self.output_variables[var_name]
        else:
            raise ValueError(f"Variabel {var_name} tidak ditemukan")
            
        x = np.linspace(var_info['min'], var_info['max'], 1000)
        
        plt.figure(figsize=(10, 6))
        for set_name, mf in var_info['sets'].items():
            y = self._membership_function(x, mf['type'], mf['params'])
            plt.plot(x, y, label=set_name, linewidth=2)
            
        plt.xlabel(var_name)
        plt.ylabel('Membership Degree')
        plt.title(f'Membership Functions for {var_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        plt.show()

# Contoh penggunaan
if __name__ == "__main__":
    # Buat sistem fuzzy Mamdani
    fuzzy_system = MamdaniFuzzySystem()
    
    # Tambahkan variabel input
    fuzzy_system.add_input_variable("temperature", 0, 100)
    fuzzy_system.add_input_variable("humidity", 0, 100)
    
    # Tambahkan variabel output
    fuzzy_system.add_output_variable("comfort", 0, 100)
    
    # Tambahkan fungsi keanggotaan untuk temperature
    fuzzy_system.add_input_membership("temperature", "cold", "triangular", [0, 0, 30])
    fuzzy_system.add_input_membership("temperature", "warm", "triangular", [20, 50, 80])
    fuzzy_system.add_input_membership("temperature", "hot", "triangular", [70, 100, 100])
    
    # Tambahkan fungsi keanggotaan untuk humidity
    fuzzy_system.add_input_membership("humidity", "low", "triangular", [0, 0, 40])
    fuzzy_system.add_input_membership("humidity", "medium", "triangular", [30, 50, 70])
    fuzzy_system.add_input_membership("humidity", "high", "triangular", [60, 100, 100])
    
    # Tambahkan fungsi keanggotaan untuk output comfort
    fuzzy_system.add_output_membership("comfort", "uncomfortable", "triangular", [0, 0, 40])
    fuzzy_system.add_output_membership("comfort", "moderate", "triangular", [30, 50, 70])
    fuzzy_system.add_output_membership("comfort", "comfortable", "triangular", [60, 100, 100])
    
    # Tambahkan rules
    fuzzy_system.add_rule([("temperature", "cold"), ("humidity", "low")], ("comfort", "comfortable"))
    fuzzy_system.add_rule([("temperature", "warm"), ("humidity", "medium")], ("comfort", "comfortable"))
    fuzzy_system.add_rule([("temperature", "hot"), ("humidity", "high")], ("comfort", "uncomfortable"))
    fuzzy_system.add_rule([("temperature", "cold"), ("humidity", "high")], ("comfort", "moderate"))
    fuzzy_system.add_rule([("temperature", "hot"), ("humidity", "low")], ("comfort", "moderate"))
    
    # Test prediksi
    test_inputs = {"temperature": 25, "humidity": 45}
    result = fuzzy_system.predict(test_inputs)
    print(f"Input: {test_inputs}")
    print(f"Output comfort level: {result:.2f}")