import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_core.tools import create_retriever_tool
import logging

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure logging for MediBlaze medical bot
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [MediBlaze] %(message)s'
)
logger = logging.getLogger(__name__)

# Configure DuckDuckGo search for medical queries only
wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
duckduckgo_search = DuckDuckGoSearchResults(api_wrapper=wrapper)

@tool
def medical_web_search(query: str) -> str:
    """
    🔍 Search the web for comprehensive medical, health, and wellness information.
    Use this for ALL health-related topics including:
    - Medical conditions, diseases, symptoms, treatments
    - Nutrition, diet, lifestyle, fitness, and wellness
    - Mental health, hygiene, preventive care
    - Health trends, research, and general health questions
    - Topics where current, diverse medical perspectives would be valuable
    This tool provides broader, up-to-date health information beyond the knowledge base.
    """
    try:        # Enhance search query for better health results
        medical_query = f"{query} health medical wellness site:who.int OR site:mayoclinic.org OR site:webmd.com OR site:healthline.com OR site:medlineplus.gov OR site:cdc.gov OR site:nih.gov"
        logger.info(f"🔍 [MediBlaze] Executing medical web search for: {query}")
        
        # Indicate search is happening
        search_indicator = "🔍 **Searching web for latest medical information...**\n\n"
        
        results = duckduckgo_search.invoke(medical_query)
        
        if not results or len(str(results).strip()) < 20:
            logger.warning("⚠️ [MediBlaze] No relevant medical web results found")
            return f"{search_indicator}I couldn't find current web information about that health topic. Please consult with a healthcare professional for the most accurate information."
        
        logger.info("✅ [MediBlaze] Medical web search completed successfully")
        return f"{search_indicator}**🌐 Latest Health Information from Web:**\n\n{results}"
        
    except Exception as e:
        logger.error(f"❌ [MediBlaze] Error during medical web search: {str(e)}")
        return "⚠️ An error occurred while searching for health information online. Please try again or consult with a healthcare professional."

@tool
def rag_tool(query: str) -> str:
    """
    📚 Retrieve relevant health information from the MediBlaze knowledge base.
    This contains comprehensive medical documents and health information covering diseases, treatments, symptoms, and wellness.
    """
    try:
        logger.info("🔧 [MediBlaze] Initializing health knowledge embeddings...")
        embeddings = PineconeEmbeddings(model="multilingual-e5-large")
        
        index_name = "mediblaze-bot"
        logger.info(f"🔗 [MediBlaze] Connecting to health knowledge base: {index_name}")
        
        # Connect to Pinecone health knowledge base
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 7})  # Get more relevant health docs
        
        _rag_tool_instance = create_retriever_tool(
            retriever,
            "search_health_knowledge_base",
            "🏥 Searches MediBlaze health knowledge base for comprehensive information about diseases, treatments, medications, symptoms, prevention, diagnosis, lifestyle health, and wellness information."
        )
        
        logger.info(f"📖 [MediBlaze] Executing health knowledge search for: {query}")
        result = _rag_tool_instance.invoke(query)        
        # If no relevant results found, provide helpful fallback
        if not result or len(str(result).strip()) < 20:
            logger.warning("⚠️ [MediBlaze] No relevant results found in health knowledge base")
            return "📚 I couldn't find specific information about that in our health knowledge base. Let me search for current health information online to help you better."
        
        logger.info("✅ [MediBlaze] Health knowledge search completed successfully")
        return f"**📚 From MediBlaze Health Knowledge Base:**\n\n{result}"
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ [MediBlaze] Error during health knowledge search: {error_msg}")
        
        # Handle missing Pinecone index gracefully
        if "NOT_FOUND" in error_msg or "not found" in error_msg.lower():
            logger.warning("⚠️ [MediBlaze] Pinecone index not found - falling back to web search only")
            return "📚 The health knowledge base is currently unavailable. I'll use web search to find current medical information for you."
        
        return "⚠️ An error occurred while searching our health knowledge base. Let me search the web for current health information instead."

@tool
def disease_prediction(symptoms: str, duration: str, severity: str, additional_info: str = "") -> str:
    """
    🔬 Analyze symptoms and predict most likely medical conditions with risk assessment.
    
    Use this tool when you have gathered enough information (after 2-3 exchanges) including:
    - Primary symptoms (fever, headache, cough, etc.)
    - Duration (how long they've had symptoms)
    - Severity (mild, moderate, severe OR pain scale 1-10)
    - Additional context (contact history, medications tried, other symptoms)
    
    This will provide:
    - Top 3-5 most likely conditions
    - Risk score for each condition
    - Specific next steps and recommendations
    - When to seek immediate care
    
    Parameters:
    - symptoms: Main symptoms described (e.g., "fever, headache behind eyes, body aches, nausea")
    - duration: How long symptoms have persisted (e.g., "2 days", "1 week")
    - severity: Intensity level (e.g., "moderate", "severe", "7/10 pain")
    - additional_info: Contact history, medications, negatives (e.g., "colleague had viral fever, took paracetamol, no cough")
    """
    try:
        logger.info(f"🔬 [MediBlaze] Analyzing symptoms for disease prediction: {symptoms}")
        
        # Use RAG to get relevant medical information
        embeddings = PineconeEmbeddings(model="multilingual-e5-large")
        vectorstore = PineconeVectorStore(
            index_name="mediblaze-bot",
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        # Enhanced query for disease prediction
        prediction_query = f"diseases conditions with symptoms: {symptoms}, duration {duration}, severity {severity}. Differential diagnosis, causes, treatment."
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Get more docs for better prediction
        docs = retriever.invoke(prediction_query)
        
        # Build knowledge context
        knowledge_context = "\n\n".join([doc.page_content for doc in docs[:5]]) if docs else "Limited information available"
        
        # Symptom severity scoring
        severity_lower = severity.lower()
        if any(term in severity_lower for term in ["severe", "worst", "unbearable", "9", "10"]):
            severity_score = "HIGH"
        elif any(term in severity_lower for term in ["moderate", "significant", "6", "7", "8"]):
            severity_score = "MODERATE"
        else:
            severity_score = "MILD"
        
        # Duration risk assessment
        duration_lower = duration.lower()
        if any(term in duration_lower for term in ["week", "weeks", "month", "chronic"]):
            duration_risk = "PROLONGED"
        elif any(term in duration_lower for term in ["days", "3 day", "4 day", "5 day"]):
            duration_risk = "MODERATE"
        else:
            duration_risk = "ACUTE"
        
        # Build prediction response
        prediction_response = f"""
🔬 **DISEASE PREDICTION ANALYSIS**

📋 **Symptoms Summary:**
• Primary: {symptoms}
• Duration: {duration} ({duration_risk})
• Severity: {severity} ({severity_score} risk)
{f"• Additional: {additional_info}" if additional_info else ""}

---

🎯 **Most Likely Conditions:**

"""
        
        # Pattern matching for common conditions
        symptoms_lower = symptoms.lower()
        
        # Initialize predictions list
        predictions = []
        
        # Fever-related conditions
        if "fever" in symptoms_lower:
            if "headache behind eyes" in symptoms_lower or "eye pain" in symptoms_lower:
                if "nausea" in symptoms_lower or "vomit" in symptoms_lower:
                    predictions.append({
                        "condition": "Dengue Fever",
                        "probability": "HIGH (75-85%)",
                        "risk": "MODERATE-HIGH",
                        "reasoning": "Classic triad: fever + retro-orbital headache + nausea. Contact history supports viral transmission.",
                        "tests": "Complete Blood Count (CBC), Dengue NS1 antigen, IgM/IgG antibodies",
                        "actions": ["See doctor within 24 hours", "Monitor platelet count", "Hydrate aggressively", "Avoid NSAIDs (aspirin/ibuprofen)"]
                    })
            
            if "body ache" in symptoms_lower or "muscle" in symptoms_lower:
                predictions.append({
                    "condition": "Viral Fever (Influenza/Common Viral Infection)",
                    "probability": "HIGH (70-80%)",
                    "risk": "MODERATE",
                    "reasoning": "Fever + body aches + systemic symptoms. Contact with sick colleague increases likelihood.",
                    "tests": "Usually clinical diagnosis, Rapid Flu test if severe",
                    "actions": ["Rest and hydration", "Paracetamol for fever", "Monitor for 48-72 hours", "See doctor if worsening"]
                })
            
            if "cough" not in symptoms_lower and "throat" not in symptoms_lower:
                predictions.append({
                    "condition": "Non-Respiratory Viral Infection",
                    "probability": "MODERATE (60-70%)",
                    "risk": "LOW-MODERATE",
                    "reasoning": "Systemic symptoms without respiratory signs suggest non-respiratory viral infection.",
                    "tests": "CBC, CRP (inflammatory markers)",
                    "actions": ["Symptomatic treatment", "Monitor temperature", "Seek care if fever >3 days"]
                })
        
        # Headache-focused conditions
        if "headache" in symptoms_lower:
            if "nausea" in symptoms_lower and ("light" in symptoms_lower or "sensitivity" in symptoms_lower):
                predictions.append({
                    "condition": "Migraine",
                    "probability": "HIGH (70-85%)",
                    "risk": "MODERATE",
                    "reasoning": "Headache + photophobia + nausea = classic migraine triad.",
                    "tests": "Clinical diagnosis, imaging if red flags present",
                    "actions": ["Dark, quiet room", "Triptans or NSAIDs (if appropriate)", "Antiemetics for nausea", "Avoid triggers"]
                })
            
            if severity_score == "HIGH" and "sudden" in symptoms_lower:
                predictions.append({
                    "condition": "⚠️ SERIOUS: Possible Meningitis/Intracranial Issue",
                    "probability": "LOW-MODERATE (15-30%)",
                    "risk": "🚨 VERY HIGH",
                    "reasoning": "Severe headache + fever + nausea can indicate serious infection.",
                    "tests": "URGENT: CT scan, Lumbar puncture",
                    "actions": ["🚨 SEEK EMERGENCY CARE IMMEDIATELY", "Do NOT delay", "Check for neck stiffness, confusion"]
                })
        
        # Respiratory conditions
        if "cough" in symptoms_lower:
            if "fever" in symptoms_lower:
                predictions.append({
                    "condition": "Lower Respiratory Tract Infection (Bronchitis/Pneumonia)",
                    "probability": "MODERATE-HIGH (60-75%)",
                    "risk": "MODERATE-HIGH",
                    "reasoning": "Productive cough + fever suggests bacterial or viral LRTI.",
                    "tests": "Chest X-ray, Sputum culture",
                    "actions": ["See doctor for evaluation", "May need antibiotics", "Monitor breathing", "Hydration"]
                })
            else:
                predictions.append({
                    "condition": "Post-Viral Cough / Upper Respiratory Infection",
                    "probability": "HIGH (70-80%)",
                    "risk": "LOW",
                    "reasoning": "Isolated cough without fever often post-viral.",
                    "tests": "Usually none needed",
                    "actions": ["Honey, steam inhalation", "OTC cough suppressants", "See doctor if >2 weeks"]
                })
        
        # GI conditions
        if any(term in symptoms_lower for term in ["stomach", "nausea", "vomit", "diarrhea", "abdominal"]):
            if "seafood" in additional_info.lower() or "food" in symptoms_lower:
                predictions.append({
                    "condition": "Food Poisoning / Gastroenteritis",
                    "probability": "HIGH (75-85%)",
                    "risk": "MODERATE",
                    "reasoning": "Recent seafood consumption + GI symptoms = likely food poisoning.",
                    "tests": "Stool culture if severe",
                    "actions": ["Aggressive hydration (ORS)", "BRAT diet", "Avoid dairy temporarily", "See doctor if bloody stool/high fever"]
                })
        
        # If no predictions yet, add generic viral illness
        if not predictions:
            predictions.append({
                "condition": "Undifferentiated Viral Illness",
                "probability": "MODERATE (50-70%)",
                "risk": "MODERATE",
                "reasoning": "Symptoms suggest viral infection but pattern unclear. Needs more clinical evaluation.",
                "tests": "CBC, CRP, viral panel if indicated",
                "actions": ["Symptomatic management", "Medical evaluation recommended", "Monitor closely"]
            })
        
        # Build response with predictions
        for idx, pred in enumerate(predictions[:5], 1):  # Top 5 predictions
            risk_emoji = "🚨" if "HIGH" in pred["risk"] else "⚠️" if "MODERATE" in pred["risk"] else "ℹ️"
            
            prediction_response += f"""
**{idx}. {pred['condition']}**
{risk_emoji} Probability: {pred['probability']} | Risk Level: {pred['risk']}

📝 **Why this diagnosis:**
{pred['reasoning']}

🧪 **Recommended Tests:**
{pred['tests']}

✅ **What You Should Do:**
"""
            for action in pred['actions']:
                prediction_response += f"\n• {action}"
            
            prediction_response += "\n\n"
        
        # Add general recommendations
        prediction_response += f"""
---

💡 **General Recommendations:**

✅ **Immediate Steps:**
• Continue monitoring temperature every 4-6 hours
• Maintain fluid intake (2-3 liters/day minimum)
• Get adequate rest (8+ hours sleep)
• Keep symptom diary (helps doctor assessment)

🏥 **When to See a Doctor:**
• Symptoms persist beyond {3 if duration_risk == "ACUTE" else 1} days without improvement
• Fever >103°F (39.4°C) or persistent fever
• Development of new concerning symptoms
• Unable to keep fluids down
• Severe weakness or confusion

🚨 **SEEK EMERGENCY CARE IF:**
• Difficulty breathing or chest pain
• Severe headache with neck stiffness
• Persistent vomiting leading to dehydration
• Altered mental status or confusion
• Symptoms of internal bleeding
• Fever with rash that doesn't blanch

---

⚠️ **Important Disclaimer:**
This is an AI-assisted prediction based on symptom patterns. It is NOT a definitive diagnosis. Only a healthcare professional can provide accurate diagnosis after proper examination and tests. If in doubt, always consult a doctor.

📞 **Need Human Medical Advice?** Contact your healthcare provider or visit nearest clinic/hospital.
"""
        
        logger.info("✅ [MediBlaze] Disease prediction analysis completed")
        return prediction_response
        
    except Exception as e:
        logger.error(f"❌ [MediBlaze] Error in disease prediction: {str(e)}")
        return f"⚠️ Unable to perform disease prediction analysis at this time. Error: {str(e)}\n\nPlease consult with a healthcare professional for proper diagnosis."