import requests

def agente_analisis_producto_ollama(descripcion, modelo='mistral'):
    prompt = f"""
Eres un experto en marketing. Resume este producto para generar un brief creativo respondiendo claramente:

1. ¿Qué es el producto?
2. ¿Qué problema resuelve?
3. ¿Cuáles son sus 3–5 características clave?
4. ¿Cuál es el principal beneficio para el usuario?
5. ¿Cuál es su diferenciador frente a la competencia?

Producto:
\"\"\"{descripcion}\"\"\"
"""

    respuesta = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': modelo,
            'prompt': prompt,
            'stream': False
        }
    )

    if respuesta.status_code == 200:
        return respuesta.json()['response']
    else:
        return f"Error: {respuesta.status_code} - {respuesta.text}"


# --- Ejemplo de uso
if __name__ == "__main__":
    descripcion_producto = """
    El SmartBottle X2 es una botella inteligente de acero inoxidable con capacidad de 600ml, que incluye sensores de temperatura y una pantalla táctil LED. 
    Se conecta vía Bluetooth con una app móvil para registrar el consumo de agua diario, enviar recordatorios y analizar hábitos de hidratación. 
    Ideal para deportistas y personas con rutinas exigentes.
    """
    
    resumen = agente_analisis_producto_ollama(descripcion_producto)
    print("📦 RESUMEN PARA MARKETING:\n")
    print(resumen)