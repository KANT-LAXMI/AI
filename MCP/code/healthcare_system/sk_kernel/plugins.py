"""
Semantic Kernel Plugins
Custom SK plugins for medical analysis and diagnosis assistance
"""

from semantic_kernel.functions import kernel_function as sk_function
from database.operations import DatabaseOperations
import json
from typing import Optional


# ===============================
# MEDICAL ANALYSIS PLUGIN
# ===============================
class MedicalAnalysisPlugin:
    """
    Plugin for analyzing medical data and generating insights.
    Integrates with the database for real patient data.
    """
    
    def __init__(self):
        self.db_ops = DatabaseOperations()
    
    @sk_function(
        description="Analyzes patient vital signs and flags any abnormal values",
        name="analyze_vitals"
    )
    async def analyze_vitals(
        self,
        blood_pressure: str,
        heart_rate: int,
        temperature: float,
        oxygen_saturation: int
    ) -> str:
        """
        Analyze vital signs and identify any concerning values.
        
        Args:
            blood_pressure: Format "systolic/diastolic" (e.g., "120/80")
            heart_rate: Heart rate in BPM
            temperature: Temperature in Fahrenheit
            oxygen_saturation: SpO2 percentage
        
        Returns:
            JSON string with analysis results
        """
        try:
            # Parse blood pressure
            systolic, diastolic = map(int, blood_pressure.split('/'))
            
            # Define normal ranges
            concerns = []
            
            # Blood pressure analysis
            if systolic > 140 or diastolic > 90:
                concerns.append({
                    "parameter": "Blood Pressure",
                    "value": blood_pressure,
                    "status": "HIGH",
                    "concern": "Hypertension risk - Stage 1 or higher"
                })
            elif systolic < 90 or diastolic < 60:
                concerns.append({
                    "parameter": "Blood Pressure",
                    "value": blood_pressure,
                    "status": "LOW",
                    "concern": "Hypotension - may cause dizziness"
                })
            
            # Heart rate analysis
            if heart_rate > 100:
                concerns.append({
                    "parameter": "Heart Rate",
                    "value": heart_rate,
                    "status": "HIGH",
                    "concern": "Tachycardia - elevated heart rate"
                })
            elif heart_rate < 60:
                concerns.append({
                    "parameter": "Heart Rate",
                    "value": heart_rate,
                    "status": "LOW",
                    "concern": "Bradycardia - low heart rate"
                })
            
            # Temperature analysis
            if temperature > 100.4:
                concerns.append({
                    "parameter": "Temperature",
                    "value": temperature,
                    "status": "HIGH",
                    "concern": "Fever present - possible infection"
                })
            elif temperature < 95.0:
                concerns.append({
                    "parameter": "Temperature",
                    "value": temperature,
                    "status": "LOW",
                    "concern": "Hypothermia risk"
                })
            
            # Oxygen saturation analysis
            if oxygen_saturation < 95:
                concerns.append({
                    "parameter": "Oxygen Saturation",
                    "value": oxygen_saturation,
                    "status": "LOW",
                    "concern": "Hypoxemia - low blood oxygen"
                })
            
            # Generate summary
            if not concerns:
                status = "NORMAL"
                message = "All vital signs are within normal ranges"
            else:
                status = "ABNORMAL"
                message = f"{len(concerns)} concerning vital sign(s) detected"
            
            return json.dumps({
                "status": status,
                "message": message,
                "concerns": concerns,
                "vitals": {
                    "blood_pressure": blood_pressure,
                    "heart_rate": heart_rate,
                    "temperature": temperature,
                    "oxygen_saturation": oxygen_saturation
                }
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @sk_function(
        description="Calculates BMI and provides weight category",
        name="calculate_bmi"
    )
    def calculate_bmi(self, weight_lbs: float, height_inches: float) -> str:
        """
        Calculate BMI and provide health category.
        
        Args:
            weight_lbs: Weight in pounds
            height_inches: Height in inches
        
        Returns:
            JSON string with BMI calculation and category
        """
        try:
            # Convert to metric
            weight_kg = weight_lbs * 0.453592
            height_m = height_inches * 0.0254
            
            # Calculate BMI
            bmi = weight_kg / (height_m ** 2)
            
            # Determine category
            if bmi < 18.5:
                category = "Underweight"
                recommendation = "Consider consulting with a nutritionist"
            elif 18.5 <= bmi < 25:
                category = "Normal weight"
                recommendation = "Maintain current healthy lifestyle"
            elif 25 <= bmi < 30:
                category = "Overweight"
                recommendation = "Consider diet and exercise modifications"
            else:
                category = "Obese"
                recommendation = "Consult healthcare provider for weight management plan"
            
            return json.dumps({
                "bmi": round(bmi, 1),
                "category": category,
                "recommendation": recommendation,
                "measurements": {
                    "weight_lbs": weight_lbs,
                    "height_inches": height_inches
                }
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @sk_function(
        description="Identifies potential risk factors based on patient data",
        name="assess_risk_factors"
    )
    def assess_risk_factors(
        self,
        age: int,
        gender: str,
        smoking: bool,
        bmi: float,
        blood_pressure: str,
        family_history: Optional[str] = None
    ) -> str:
        """
        Assess cardiovascular and general health risk factors.
        
        Returns:
            JSON string with risk assessment
        """
        try:
            risk_factors = []
            risk_score = 0
            
            # Age risk
            if age > 65:
                risk_factors.append("Advanced age (>65)")
                risk_score += 2
            elif age > 50:
                risk_factors.append("Age over 50")
                risk_score += 1
            
            # Smoking
            if smoking:
                risk_factors.append("Current smoker")
                risk_score += 3
            
            # BMI
            if bmi >= 30:
                risk_factors.append("Obesity (BMI ≥30)")
                risk_score += 2
            elif bmi >= 25:
                risk_factors.append("Overweight (BMI 25-30)")
                risk_score += 1
            
            # Blood pressure
            try:
                systolic, diastolic = map(int, blood_pressure.split('/'))
                if systolic > 140 or diastolic > 90:
                    risk_factors.append("Hypertension")
                    risk_score += 2
            except:
                pass
            
            # Family history
            if family_history:
                risk_factors.append(f"Family history: {family_history}")
                risk_score += 1
            
            # Determine overall risk level
            if risk_score >= 6:
                risk_level = "HIGH"
                recommendation = "Schedule comprehensive health evaluation immediately"
            elif risk_score >= 3:
                risk_level = "MODERATE"
                recommendation = "Schedule preventive care appointment within 1 month"
            else:
                risk_level = "LOW"
                recommendation = "Continue routine health maintenance"
            
            return json.dumps({
                "risk_level": risk_level,
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "recommendation": recommendation
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})


# ===============================
# DIAGNOSIS ASSISTANT PLUGIN
# ===============================
class DiagnosisAssistantPlugin:
    """
    Plugin to assist with differential diagnosis based on symptoms.
    This is for educational/assistive purposes only - not for actual diagnosis.
    """
    
    @sk_function(
        description="Generates differential diagnosis list based on symptoms",
        name="differential_diagnosis"
    )
    def differential_diagnosis(self, symptoms: str, age: int, gender: str) -> str:
        """
        Generate a list of possible conditions based on symptoms.
        
        IMPORTANT: This is for educational purposes only. Always consult a healthcare provider.
        
        Args:
            symptoms: Comma-separated list of symptoms
            age: Patient age
            gender: Patient gender
        
        Returns:
            JSON string with possible conditions
        """
        # This is a simplified rule-based system
        # In production, integrate with medical knowledge bases or APIs
        
        symptoms_lower = symptoms.lower()
        possible_conditions = []
        
        # Simple pattern matching (demonstration only)
        if "fever" in symptoms_lower and "cough" in symptoms_lower:
            possible_conditions.append({
                "condition": "Upper Respiratory Infection",
                "likelihood": "Common",
                "key_features": ["fever", "cough", "congestion"]
            })
            possible_conditions.append({
                "condition": "Influenza",
                "likelihood": "Moderate",
                "key_features": ["fever", "cough", "body aches", "fatigue"]
            })
        
        if "chest pain" in symptoms_lower:
            if age > 40:
                possible_conditions.append({
                    "condition": "Cardiac Event (requires immediate evaluation)",
                    "likelihood": "Must rule out",
                    "key_features": ["chest pain", "shortness of breath", "arm pain"]
                })
            possible_conditions.append({
                "condition": "Costochondritis",
                "likelihood": "Moderate",
                "key_features": ["chest wall pain", "tender to touch"]
            })
        
        if "headache" in symptoms_lower and "stiff neck" in symptoms_lower:
            possible_conditions.append({
                "condition": "Meningitis (requires immediate evaluation)",
                "likelihood": "Must rule out",
                "key_features": ["severe headache", "stiff neck", "fever", "photophobia"]
            })
        
        if "abdominal pain" in symptoms_lower:
            possible_conditions.append({
                "condition": "Gastroenteritis",
                "likelihood": "Common",
                "key_features": ["abdominal pain", "nausea", "diarrhea"]
            })
            if age > 50:
                possible_conditions.append({
                    "condition": "Diverticulitis",
                    "likelihood": "Consider",
                    "key_features": ["left lower abdominal pain", "fever"]
                })
        
        if not possible_conditions:
            possible_conditions.append({
                "condition": "Insufficient information for differential",
                "likelihood": "N/A",
                "key_features": []
            })
        
        return json.dumps({
            "disclaimer": "⚠️ This is an AI-generated differential diagnosis for educational purposes only. Always consult a licensed healthcare provider for proper diagnosis and treatment.",
            "patient_info": {
                "age": age,
                "gender": gender,
                "symptoms": symptoms
            },
            "possible_conditions": possible_conditions,
            "recommendations": [
                "Seek immediate medical attention if symptoms worsen",
                "Document symptom progression",
                "Schedule appointment with healthcare provider",
                "Call emergency services if experiencing severe symptoms"
            ]
        }, indent=2)
    
    @sk_function(
        description="Suggests appropriate medical tests based on symptoms and suspected conditions",
        name="suggest_tests"
    )
    def suggest_tests(self, suspected_condition: str, symptoms: str) -> str:
        """
        Suggest appropriate diagnostic tests.
        
        Args:
            suspected_condition: The condition being investigated
            symptoms: Patient symptoms
        
        Returns:
            JSON string with recommended tests
        """
        condition_lower = suspected_condition.lower()
        recommended_tests = []
        
        # Map conditions to tests (simplified)
        test_mapping = {
            "diabetes": ["Fasting blood glucose", "HbA1c", "Oral glucose tolerance test"],
            "hypertension": ["Blood pressure monitoring", "EKG", "Echocardiogram", "Blood tests"],
            "infection": ["Complete blood count", "Blood cultures", "Urinalysis", "Chest X-ray"],
            "cardiac": ["EKG", "Troponin levels", "Echocardiogram", "Stress test"],
            "respiratory": ["Chest X-ray", "Spirometry", "Pulse oximetry", "Sputum culture"],
            "anemia": ["Complete blood count", "Iron studies", "Vitamin B12", "Folate"],
        }
        
        # Find relevant tests
        for key, tests in test_mapping.items():
            if key in condition_lower:
                recommended_tests.extend(tests)
        
        if not recommended_tests:
            recommended_tests = ["Complete blood count", "Basic metabolic panel", "Urinalysis"]
        
        return json.dumps({
            "suspected_condition": suspected_condition,
            "recommended_tests": list(set(recommended_tests)),  # Remove duplicates
            "note": "These are suggested tests. A healthcare provider will determine appropriate testing based on clinical judgment."
        }, indent=2)