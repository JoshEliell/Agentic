from django.shortcuts import render
from django.http import HttpResponse
from .forms import AnswerForm, FileInputForm
import json
# utils.py
import tempfile
from pptx import Presentation
import fitz  # PyMuPDF
import subprocess
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
#Agentics
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Create your views here.
def home(request):
    resultados = None  # Inicializa variable
    preguntas = None
    structured_objective = None
    
    if request.method == 'POST':
        form = FileInputForm(request.POST, request.FILES)
        if form.is_valid():
            tipo = form.cleaned_data['tipo']
            archivo = request.FILES['archivo']
            resultados = analizar_archivo(tipo, archivo)            
            campaign_objectives = resultados.get("Campaign objectives", [])
            product_specifications = resultados.get("Product or service specifications", [])
            audience_data = resultados.get("Audience data or segments", [])
            content_type = resultados.get("Type of content to generate", [])
            tone_style = resultados.get("Preferred tone and style", [])
            historical_context = resultados.get("Previous information / historical context", [])
            business_insights = resultados.get("Business insights or findings", [])
            stakeholder_comments = resultados.get("Internal stakeholder comments", [])
            content_tags = resultados.get("Content taxonomy or tags", [])
            delivery_tools = resultados.get("Expected delivery tools/platforms", [])

            campaign_objectives_txt = "\n".join(campaign_objectives)
            product_specifications_txt = "\n".join(product_specifications)
            audience_data_txt = "\n".join(audience_data)
            content_type_txt = "\n".join(content_type)
            tone_style_txt = "\n".join(tone_style)
            historical_context_txt = "\n".join(historical_context)
            agent_1 = agentic_objetive(campaign_objectives_txt)
            agent_2 = agentic_product_analysis(product_specifications_txt, audience_data_txt, content_type_txt, tone_style_txt)
            agent_3 = agentic_background(historical_context_txt)
            agent_4 = agentic_audience_profiling(audience_data_txt, agent_2)
            agent_5 = agentic_insights(product_specifications_txt, campaign_objectives_txt, agent_4, agent_2)
            # Final brief using all the relevant data
            final_brief = agentic_brief_generator(product_specifications_txt,agent_4,agent_1,agent_5,agent_3)
            preguntas = final_brief
    else:
        form = FileInputForm()

    context= {'form': form,
              'resultados': resultados,
              'preguntas': preguntas,
              'structured_objective':structured_objective,
        }

    return render(request, 'dashboard/dashboard.html',context)

def analizar_archivo(tipo, archivo):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for chunk in archivo.chunks():
            tmp.write(chunk)
        ruta = tmp.name

    texto = extraer_texto(tipo, ruta)
    result_str = extract_insights_with_rag(texto)
    try:
        return json.loads(result_str)
    except json.JSONDecodeError as e:
        print("❌ Error al decodificar JSON:", e)
        return {}

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

    # 4. Define retrievera
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

def agentic_objetive(objective: str, model_name: str = "mistral") -> str: #1
    llm = OllamaLLM(model=model_name)

    template = """
    You are an AI agent that receives a business campaign objective and converts it into structured key variables.

    Answer the following in a structured way:
    - Business Objective
    - What are we trying to achieve?
    - Why is this campaign important to the business?
    - Key business KPIs

    Objective:
    {objective}
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    result = chain.invoke({"objective": objective})

    return result

def agentic_product_analysis(product_specifications: str,audience_data: str,content_type: str,tone_style: str, model_name: str = "mistral") -> str: #2
    llm = OllamaLLM(model=model_name)

    template = """
    You are a marketing expert. Analyze the product below to help answer the following:

    1. Marketing Objective
    2. The Problem we are trying to solve
    3. What are the challenges?
    4. Product/Service Description
    5. Solutions/Offering
    6. Why this platform or solution (XYZ)?
    7. Why does the enterprise need it?
    8. 3–5 key features
    9. Main user benefit
    10. What makes it different from competitors?

    ---

    Product description:
    {description}
    Audience:
    {audience_data}
    Type of content:
    {content_type}
    Tone style:
    {tone_style}
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    result = chain.invoke({"description": product_specifications,"audience_data": audience_data,"content_type": content_type,"tone_style": tone_style,})

    return result

def agentic_background(historical_context: str, model_name: str = "mistral") -> str: #3
    llm = OllamaLLM(model=model_name)

    template = """
    You are a meticulous marketing analyst specializing in campaign retrospectives.
    Your task is to carefully review the provided historical campaign data.
    Based on this data, extract and summarize the most critical takeaways.

    Focus on identifying:
    1.  **Key Successes:** What specific aspects or elements of past campaigns worked exceptionally well?
    2.  **Key Failures or Challenges:** What didn't work as expected, what were the main obstacles, or what valuable lessons were learned from setbacks?
    3.  **Notable Creative Themes/Approaches:** Were there any distinct creative styles, messaging tones, or visual approaches that were particularly effective or ineffective?
    4.  **Overall Strategic Implications:** How should these past learnings inform future marketing decisions, strategy adjustments, or avoid repeating mistakes?

    ---

    Past Campaign Data:
    \"\"\"{historical_context}\"\"\"
    ---
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    result = chain.invoke({"historical_context": historical_context})

    return result

def agentic_audience_profiling(audience_data: str,agent_2: str, model_name: str = "mistral") -> str: #4
    llm = OllamaLLM(model=model_name)

    template = """
    You are a digital strategist. Use the audience data to answer:

    1. Target Audience
    2. Ideal Digital Mediums/Channels for this audience
    3. Tone and Style of communication
    4. Key behavioral patterns
    5. Platform preferences

    Audience data:
    {audience_data}
    Product Analysis:
    {product_analysis}
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    result = chain.invoke({"audience_data": audience_data,"product_analysis": agent_2})

    return result

def agentic_insights(product_specifications: str,campaign_objectives: str,agent_4: str,agent_2: str,model_name: str = "mistral") -> str: #5
    llm = OllamaLLM(model=model_name)

    template = """
    You are a strategic planner.

    Analyze the data and provide:

    1. Strategic Insight (linking product + audience + objective)
    2. Creative idea based on that insight
    3. Present market trend and demand
    4. What makes this moment relevant for this campaign

    Product:
    {product}
    Audience:
    {audience}
    Campaign Objective:
    {campaign_objectives}
    Product Analysis:
    {product_analysis}
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    # Pass all three variables to the invoke method
    result = chain.invoke({
        "product": product_specifications,
        "campaign_objectives": campaign_objectives,
        "audience": agent_4,
        "product_analysis": agent_2,
    })

    return result

def agentic_brief_generator(product_specifications: str,agent_4: str,agent_1: str,agent_5: str,agent_3: str,model_name: str = 'mistral') -> str:
    llm = OllamaLLM(model=model_name)

    template = """
    You are a creative strategist at a marketing agency. Build a structured campaign brief with the following sections:

    1. Brief Title
    2. Business Objective
    3. Marketing Objective
    4. Background
    5. Target Audience
    6. The Problem we are trying to solve
    7. What are the challenges?
    8. Product/Service Description
    9. Strategic Insight
    10. Creative Proposal
    11. Present Market Trend and Demand
    12. Digital Channels
    13. Why XYZ (Platform)?
    14. Why does the enterprise need this solution?
    15. Agency Statement of Work (SOW)

    Product:
    {product}

    Audience:
    {audience}

    Objective:
    {objective}

    Insight:
    {insight}

    Background:
    {background}
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    # Invoke the chain with all necessary variables
    result = chain.invoke({
        "product": product_specifications,
        "audience": agent_4,
        "objective": agent_1,
        "insight": agent_5,
        "background": agent_3,
    })

    return result