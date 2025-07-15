from django.shortcuts import render
from django.http import HttpResponse
from .forms import AnswerForm, FileInputForm
# utils.py
import tempfile
from pptx import Presentation
import fitz  # PyMuPDF
import subprocess

# Create your views here.
def home(request):
    resultados = None  # Inicializa variable
    if request.method == 'POST':
        form = FileInputForm(request.POST, request.FILES)
        if form.is_valid():
            tipo = form.cleaned_data['tipo']
            archivo = request.FILES['archivo']
            resultados = analizar_archivo(tipo, archivo)
    else:
        form = FileInputForm()

    context= {'form': form,
              'resultados': resultados,
        }

    return render(request, 'dashboard/dashboard.html',context)

def analizar_archivo(tipo, archivo):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in archivo.chunks():
            tmp.write(chunk)
        ruta = tmp.name

    texto = extraer_texto(tipo, ruta)
    return extraer_insights_con_ollama(texto)

def extraer_texto(tipo, ruta):
    if tipo == 'pdf':
        doc = fitz.open(ruta)
        return "\n".join([page.get_text() for page in doc])
    
    elif tipo == 'powerpoint':
        prs = Presentation(ruta)
        texto = ''
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    texto += shape.text + '\n'
        return texto
    
    elif tipo == 'google_docs' or tipo == 'notion':
        return "Contenido simulado. Debes integrar API de Google Docs o Notion."

    return "No se pudo leer el archivo."

def extraer_insights_con_ollama(texto):
    prompt = f"""
    You are a marketing expert assistant. Based on the following text, extract the following key inputs in JSON format:
    
    1. Campaign objectives
    2. Product or service specifications
    3. Audience data or segments
    4. Type of content to generate
    5. Preferred tone and style
    6. Previous information / historical context
    7. Business insights or findings
    8. Internal stakeholder comments
    9. Content taxonomy or tags
    10. Expected delivery tools/platforms

    Reference text:
    \"\"\"
    {texto}
    \"\"\"
    """
    comando = [r'C:\Users\victo\AppData\Local\Programs\Ollama\ollama.exe', 'run', 'mistral', prompt]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    return resultado.stdout  # Puede ser JSON, texto plano o lista, según el modelo