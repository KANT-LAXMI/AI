"""
Semantic Kernel Setup
Initializes and configures Semantic Kernel with medical plugins
"""

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from config.settings import OPENAI_CONFIG, OLLAMA_CONFIG
from typing import Optional, Dict, Any
import os


# ===============================
# KERNEL INITIALIZATION
# ===============================
def setup_kernel(use_openai: bool = False) -> sk.Kernel:
    """
    Initialize and configure Semantic Kernel with appropriate LLM.
    
    Args:
        use_openai: If True, use OpenAI. If False, use Ollama (default)
    
    Returns:
        Configured Semantic Kernel instance
    """
    # Create kernel
    kernel = sk.Kernel()
    
    if use_openai and OPENAI_CONFIG["api_key"]:
        # ===============================
        # OPENAI CONFIGURATION
        # ===============================
        service_id = "openai_chat"
        kernel.add_service(
            OpenAIChatCompletion(
                service_id=service_id,
                ai_model_id=OPENAI_CONFIG["model"],
                api_key=OPENAI_CONFIG["api_key"]
            )
        )
        print(f"✅ Semantic Kernel initialized with OpenAI ({OPENAI_CONFIG['model']})")
    else:
        # ===============================
        # OLLAMA CONFIGURATION (DEFAULT)
        # ===============================
        service_id = "ollama_chat"
        kernel.add_service(
            OllamaChatCompletion(
                service_id=service_id,
                ai_model_id=OLLAMA_CONFIG["model"],
                url=OLLAMA_CONFIG["base_url"]
            )
        )
        print(f"✅ Semantic Kernel initialized with Ollama ({OLLAMA_CONFIG['model']})")
    
    return kernel


# ===============================
# KERNEL QUERY FUNCTIONS
# ===============================
async def get_kernel_response(
    kernel: sk.Kernel,
    prompt: str,
    context_variables: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get a response from Semantic Kernel with given prompt and context.
    
    Args:
        kernel: Initialized Semantic Kernel instance
        prompt: The prompt to send to the LLM
        context_variables: Optional context variables for the prompt
    
    Returns:
        String response from the LLM
    """
    try:
        # Create a semantic function from the prompt
        semantic_function = kernel.create_semantic_function(
            prompt_template=prompt,
            max_tokens=OLLAMA_CONFIG["max_tokens"],
            temperature=OLLAMA_CONFIG["temperature"]
        )
        
        # Prepare context
        context = kernel.create_new_context()
        if context_variables:
            for key, value in context_variables.items():
                context[key] = value
        
        # Execute the function
        result = await semantic_function.invoke_async(context=context)
        
        return str(result)
        
    except Exception as e:
        return f"Error getting kernel response: {str(e)}"


# ===============================
# SPECIALIZED SK FUNCTIONS
# ===============================
async def analyze_symptoms(kernel: sk.Kernel, symptoms: str, patient_history: str = "") -> str:
    """
    Use SK to analyze symptoms and provide medical insights.
    
    Args:
        kernel: Initialized Semantic Kernel
        symptoms: Patient's symptoms description
        patient_history: Patient's medical history (optional)
    
    Returns:
        Analysis and recommendations
    """
    prompt = f"""
You are a medical assistant AI helping to analyze patient symptoms.

PATIENT SYMPTOMS:
{symptoms}

PATIENT HISTORY:
{patient_history if patient_history else "No previous medical history provided"}

Please provide:
1. Possible conditions (differential diagnosis)
2. Recommended tests or examinations
3. General recommendations (NOT a medical diagnosis)
4. When to seek immediate medical attention

IMPORTANT: You are an AI assistant, not a doctor. Always recommend consulting with a healthcare professional.

Your analysis:
"""
    
    return await get_kernel_response(kernel, prompt)


async def generate_medical_summary(kernel: sk.Kernel, medical_records: str) -> str:
    """
    Generate a comprehensive medical summary from records.
    
    Args:
        kernel: Initialized Semantic Kernel
        medical_records: JSON or text of medical records
    
    Returns:
        Summarized medical history
    """
    prompt = f"""
You are a medical documentation specialist. Generate a concise but comprehensive summary of the following medical records.

MEDICAL RECORDS:
{medical_records}

Create a summary including:
1. Key diagnoses and conditions
2. Treatment history
3. Current medications
4. Notable findings or trends
5. Recommendations for ongoing care

Medical Summary:
"""
    
    return await get_kernel_response(kernel, prompt)


async def suggest_treatment_plan(
    kernel: sk.Kernel,
    diagnosis: str,
    patient_info: str,
    allergies: str = "None"
) -> str:
    """
    Suggest a treatment plan based on diagnosis and patient information.
    
    Args:
        kernel: Initialized Semantic Kernel
        diagnosis: The diagnosis or condition
        patient_info: Patient information (age, gender, conditions, etc.)
        allergies: Known allergies
    
    Returns:
        Suggested treatment plan
    """
    prompt = f"""
You are a healthcare AI assistant helping to formulate treatment plans.

DIAGNOSIS: {diagnosis}

PATIENT INFORMATION:
{patient_info}

KNOWN ALLERGIES: {allergies}

Based on the above information, suggest a comprehensive treatment plan including:
1. Recommended medications (with dosage considerations)
2. Lifestyle modifications
3. Follow-up schedule
4. Warning signs to watch for
5. Patient education points

IMPORTANT: This is a suggested plan for review by a licensed healthcare provider. Always emphasize the need for professional medical consultation.

Treatment Plan:
"""
    
    return await get_kernel_response(kernel, prompt)


async def check_medication_safety(
    kernel: sk.Kernel,
    medication: str,
    patient_conditions: str,
    current_medications: str
) -> str:
    """
    Check medication safety considering patient conditions and current medications.
    
    Args:
        kernel: Initialized Semantic Kernel
        medication: Medication being considered
        patient_conditions: Patient's medical conditions
        current_medications: Current medications
    
    Returns:
        Safety assessment
    """
    prompt = f"""
You are a pharmaceutical safety AI assistant.

MEDICATION BEING CONSIDERED: {medication}

PATIENT CONDITIONS:
{patient_conditions}

CURRENT MEDICATIONS:
{current_medications}

Provide a safety assessment including:
1. Contraindications based on patient conditions
2. Potential drug-drug interactions
3. Dosing considerations
4. Monitoring recommendations
5. Patient counseling points

Safety Assessment:
"""
    
    return await get_kernel_response(kernel, prompt)


# ===============================
# EXAMPLE USAGE
# ===============================
async def main():
    """Example of using Semantic Kernel functions"""
    # Initialize kernel
    kernel = setup_kernel(use_openai=False)  # Use Ollama
    
    # Example 1: Analyze symptoms
    print("\n" + "="*80)
    print("EXAMPLE 1: Symptom Analysis")
    print("="*80)
    
    symptoms = "Persistent cough for 2 weeks, mild fever (100.5°F), fatigue, slight chest discomfort"
    patient_history = "45-year-old male, smoker, no significant medical history"
    
    analysis = await analyze_symptoms(kernel, symptoms, patient_history)
    print(analysis)
    
    # Example 2: Generate medical summary
    print("\n" + "="*80)
    print("EXAMPLE 2: Medical Summary")
    print("="*80)
    
    records = """
    Visit 1 (2024-01-15): Hypertension diagnosed, prescribed Lisinopril 10mg daily
    Visit 2 (2024-02-20): Blood pressure controlled, continue medication
    Visit 3 (2024-03-10): Patient reports dizziness, reduced dosage to 5mg
    """
    
    summary = await generate_medical_summary(kernel, records)
    print(summary)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())