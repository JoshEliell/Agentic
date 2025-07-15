import requests

def agente_feedback_brief(brief_original, comentarios, modelo='mistral'):
    prompt = f"""
Actúa como un editor creativo. Has recibido un brief de campaña ya redactado, junto con comentarios del equipo de marketing.

Tu tarea es:
- Modificar solo las partes necesarias del brief según el feedback.
- Mantener la estructura original.
- No rehacer el brief desde cero, solo ajustar lo necesario.
- Asegúrate de que el resultado siga siendo profesional y claro.

---
📄 Brief original:
\"\"\"{brief_original}\"\"\"

💬 Comentarios del equipo:
\"\"\"{comentarios}\"\"\"

---
📄 Devuelve el brief modificado, resaltando claramente los cambios realizados.
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
    
brief_original = """
1. Título del Brief: “Domina tu hidratación” – Campaña SmartBottle X2

2. Objetivo: Posicionar SmartBottle X2 como herramienta clave en la rutina saludable de jóvenes profesionales.

3. Producto: Botella inteligente con sensores, pantalla LED y app de seguimiento de hidratación.

4. Audiencia: Jóvenes de 25–34 años, activos y preocupados por su bienestar. Usan redes como TikTok e Instagram.

5. Insight: Quieren sentirse en control de su salud.

6. Idea creativa: Reto de 21 días “Domina tu hidratación” con influencers, app y logros compartibles.
"""

comentarios = """
- El objetivo debe incluir la intención de aumentar descargas de la app.
- El tono del insight debe ser más emocional, no tan funcional.
- Incluir una recomendación de lenguaje visual en la idea creativa.
"""

resultado = agente_feedback_brief(brief_original, comentarios)
print("📘 BRIEF AJUSTADO CON FEEDBACK:\n")
print(resultado)