import requests

def agente_perfilado_audiencia_ollama(datos_audiencia, modelo='mistral'):
    prompt = f"""
Actúa como un estratega de marketing digital. Recibe los siguientes datos de audiencia y genera un perfil completo que incluya:

1. Resumen del público objetivo (edad, intereses, patrones de comportamiento)
2. Plataformas ideales para impactar a esta audiencia (redes sociales, email, etc.)
3. Estilo de comunicación sugerido (tono, tipo de mensaje, visuales)
4. Insight clave para conectar con este segmento

Audiencia:
\"\"\"{datos_audiencia}\"\"\"
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


# --- Ejemplo de uso
if __name__ == "__main__":
    datos = """
    Audiencia de entre 25 y 34 años, mayormente urbanos, con intereses en fitness, tecnología y alimentación saludable. 
    Frecuentan Instagram, YouTube y TikTok. Suelen interactuar con contenido visual, reseñas de productos y retos virales. 
    Valoran marcas auténticas y sostenibles. Compran en línea y leen reseñas antes de decidir.
    """
    
    perfil = agente_perfilado_audiencia_ollama(datos)
    print("🎯 PERFIL DE AUDIENCIA:\n")
    print(perfil)
