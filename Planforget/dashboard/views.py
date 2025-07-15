from django.shortcuts import render
from django.http import HttpResponse
from .forms import AnswerForm, FileInputForm
# utils.py
import tempfile
from pptx import Presentation
import fitz  # PyMuPDF
import subprocess
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

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
    return extract_insights_with_rag(texto)

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

# Define the RAG function
def extract_insights_with_rag(text):
    # 1. Chunk the document
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    # 2. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Vector store
    db = FAISS.from_documents(docs, embeddings)

    # 4. Define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

    # 5. Use Ollama LLM
    llm = Ollama(model="mistral")  # o "llama3", si tienes otro modelo cargado

    # 6. Define the prompt/question
    query = """
    Extract the following key inputs as JSON:
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
    """

    # 7. Retrieval-based QA chain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    result = rag_chain.run(query)

    return result