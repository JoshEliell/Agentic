from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# ✅ Nueva forma moderna de usar Ollama
llm = OllamaLLM(model="mistral")

# Prompt estructurado
template = """
Eres un agente de IA que recibe un objetivo de campaña de negocio y lo convierte en variables clave estructuradas.

Ejemplo:
Objetivo: Aumentar el reconocimiento de marca entre adultos jóvenes en TikTok.
Salida:
- Meta: Reconocimiento de marca
- Audiencia principal: Adultos jóvenes (18-25)
- Canal: TikTok
- Métricas clave: Alcance, impresiones, tasa de engagement

---

Objetivo: {objetivo}

Devuelve solo las variables clave.
"""

prompt = PromptTemplate.from_template(template)

# ✅ Nueva forma de encadenar prompt y modelo
chain = prompt | llm

# Prueba
objetivo_usuario = "Mejorar la conversión en el sitio web desde campañas de Instagram dirigidas a mujeres emprendedoras"
respuesta = chain.invoke({"objetivo": objetivo_usuario})

print("📌 Variables clave detectadas:\n")
print(respuesta)
