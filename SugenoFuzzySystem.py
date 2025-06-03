import numpy as np

class SugenoFuzzySystem:
    def __init__(self):
        self.rules = []
        self.input_variables = {}
        self.output_variables = {}
        
    def add_input_variable(self, name, min_val, max_val):
        """Menambahkan variabel input"""
        self.input_variables[name] = {'min': min_val, 'max': max_val, 'sets': {}}
        
    def add_output_variable(self, name, default_value=0):
        """Menambahkan variabel output untuk Sugeno"""
        self.output_variables[name] = {'default': default_value}
        
    def add_input_membership(self, var_name, set_name, mf_type, params):
        """Menambahkan fungsi keanggotaan untuk variabel input"""
        if var_name not in self.input_variables:
            raise ValueError(f"Variabel input {var_name} tidak ditemukan")
            
        self.input_variables[var_name]['sets'][set_name] = {
            'type': mf_type,
            'params': params
        }
        
    def add_rule(self, antecedents, consequent_value):
        """Menambahkan rule fuzzy untuk Sugeno
        antecedents: list of tuples [(var_name, set_name), ...]
        consequent_value: nilai konstan atau fungsi
        """
        self.rules.append({'if': antecedents, 'then': consequent_value})
        
    def _membership_function(self, x, mf_type, params):
        """Menghitung nilai keanggotaan"""
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
        """Inferensi menggunakan metode Sugeno"""
        rule_strengths = []
        consequent_values = []
        
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
            rule_strengths.append(rule_strength)
            
            # Untuk Sugeno, consequent adalah nilai konstan
            consequent_value = rule['then']
            if callable(consequent_value):
                # Jika consequent adalah fungsi, evaluasi dengan input asli
                input_values = {k: list(v.keys())[0] for k, v in fuzzified_inputs.items()}
                consequent_value = consequent_value(input_values)
            
            consequent_values.append(consequent_value)
            
        return rule_strengths, consequent_values
        
    def defuzzify(self, rule_strengths, consequent_values):
        """Defuzzifikasi menggunakan weighted average"""
        if not rule_strengths or not consequent_values:
            output_var = list(self.output_variables.keys())[0] if self.output_variables else None
            return self.output_variables[output_var]['default'] if output_var else 0
            
        numerator = sum(w * z for w, z in zip(rule_strengths, consequent_values) if w > 0)
        denominator = sum(w for w in rule_strengths if w > 0)
        
        if denominator == 0:
            output_var = list(self.output_variables.keys())[0] if self.output_variables else None
            return self.output_variables[output_var]['default'] if output_var else 0
            
        return numerator / denominator
        
    def predict(self, inputs):
        """Prediksi output untuk input tertentu"""
        try:
            fuzzified = self.fuzzify(inputs)
            rule_strengths, consequent_values = self.infer(fuzzified)
            return self.defuzzify(rule_strengths, consequent_values)
        except Exception as e:
            print(f"Error dalam prediksi: {e}")
            return 0

# Contoh penggunaan
if __name__ == "__main__":
    # Buat sistem fuzzy
    fuzzy_system = SugenoFuzzySystem()
    
    # Tambahkan variabel input
    fuzzy_system.add_input_variable("temperature", 0, 100)
    fuzzy_system.add_input_variable("humidity", 0, 100)
    
    # Tambahkan variabel output
    fuzzy_system.add_output_variable("comfort", 50)
    
    # Tambahkan fungsi keanggotaan untuk temperature
    fuzzy_system.add_input_membership("temperature", "cold", "triangular", [0, 0, 30])
    fuzzy_system.add_input_membership("temperature", "warm", "triangular", [20, 50, 80])
    fuzzy_system.add_input_membership("temperature", "hot", "triangular", [70, 100, 100])
    
    # Tambahkan fungsi keanggotaan untuk humidity
    fuzzy_system.add_input_membership("humidity", "low", "triangular", [0, 0, 40])
    fuzzy_system.add_input_membership("humidity", "medium", "triangular", [30, 50, 70])
    fuzzy_system.add_input_membership("humidity", "high", "triangular", [60, 100, 100])
    
    # Tambahkan rules
    fuzzy_system.add_rule([("temperature", "cold"), ("humidity", "low")], 80)
    fuzzy_system.add_rule([("temperature", "warm"), ("humidity", "medium")], 90)
    fuzzy_system.add_rule([("temperature", "hot"), ("humidity", "high")], 30)
    fuzzy_system.add_rule([("temperature", "cold"), ("humidity", "high")], 60)
    fuzzy_system.add_rule([("temperature", "hot"), ("humidity", "low")], 40)
    
    # Test prediksi
    test_inputs = {"temperature": 25, "humidity": 45}
    result = fuzzy_system.predict(test_inputs)
    print(f"Input: {test_inputs}")
    print(f"Output comfort level: {result:.2f}")