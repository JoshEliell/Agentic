import requests

def agente_insights(producto, audiencia, objetivo, modelo='mistral'):
    prompt = f"""
Actúa como un planner estratégico de marketing.

Tu tarea es analizar la siguiente información y generar un **insight estratégico** que conecte:

- El producto (qué ofrece, sus beneficios)
- La audiencia (quién es, qué busca)
- El objetivo de campaña (qué se quiere lograr)

Luego, sugiere una **idea o enfoque creativo** basado en ese insight que pueda inspirar un brief de campaña.

--- 
🧱 Producto:
{producto}

🎯 Objetivo:
{objetivo}

👥 Audiencia:
{audiencia}

---
Devuelve:
1. Insight estratégico (máx. 3 líneas)
2. Propuesta creativa basada en el insight
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
resumen_producto = """
El SmartBottle X2 es una botella inteligente con sensores de temperatura, pantalla LED y app para seguimiento de hidratación. Pensada para personas activas que quieren controlar su consumo de agua fácilmente.
"""

perfil_audiencia = """
Personas de 25-34 años, interesadas en fitness y bienestar. Usan Instagram y TikTok, valoran marcas auténticas y sostenibles, y siguen retos o recomendaciones de influencers.
"""

objetivo_campaña = "Posicionar la SmartBottle X2 como herramienta diaria para una vida saludable y consciente."

resultado = agente_insights(resumen_producto, perfil_audiencia, objetivo_campaña)
print("🔍 INSIGHT ESTRATÉGICO Y PROPUESTA:\n")
print(resultado)