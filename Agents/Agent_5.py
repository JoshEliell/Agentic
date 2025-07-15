import requests

def agente_generador_brief(producto, audiencia, objetivo, insight, idea_creativa, modelo='mistral'):
    prompt = f"""
Eres un estratega creativo de una agencia de marketing. Toma los siguientes elementos y genera un brief de campaña estructurado y profesional. El lenguaje debe ser claro, conciso y adecuado para compartir con un equipo creativo.

---
🧱 Producto:
{producto}

👥 Audiencia:
{audiencia}

🎯 Objetivo:
{objetivo}

🔍 Insight:
{insight}

💡 Propuesta Creativa:
{idea_creativa}

---

Devuelve el brief con secciones claras, como:

1. Título del Brief  
2. Objetivo de la campaña  
3. Descripción del producto/servicio  
4. Audiencia objetivo  
5. Insight estratégico  
6. Propuesta creativa  
7. Recomendaciones adicionales (opcional)
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"❌ Error {response.status_code}: {response.text}"
producto = "SmartBottle X2: botella inteligente con sensores y app para rastrear hidratación, ideal para personas activas y saludables."

audiencia = "Personas de 25–34 años interesadas en fitness, bienestar y tecnología. Consumen contenido visual, siguen retos virales y valoran marcas auténticas."

objetivo = "Aumentar awareness y adopción inicial del producto entre jóvenes profesionales activos."

insight = "Las personas que buscan bienestar también quieren sentirse en control de su salud diaria."

idea_creativa = "Campaña 'Domina tu hidratación': un reto de 21 días con influencers mostrando cómo la usan, integrando una app con estadísticas y logros."

brief = agente_generador_brief(producto, audiencia, objetivo, insight, idea_creativa)
print("📝 BRIEF GENERADO:\n")
print(brief)