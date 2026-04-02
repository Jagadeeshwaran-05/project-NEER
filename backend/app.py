from flask import Flask, jsonify, request
from flask_cors import CORS
import ee
import json
from datetime import datetime
import os
import urllib.request
import urllib.error

try:
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
    LANGCHAIN_IMPORT_ERROR = None
except Exception as import_error:
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_IMPORT_ERROR = str(import_error)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEOJSON_DIR = os.path.join(BASE_DIR, "geojson_files")

LAKE_GEOJSON_FILES = {
    "Ukkadam": "ukkadamlakepolygonmap.geojson",
    "Valankulam": "valankulam(includes chinna kulam).geojson",
    "Kurichi": "Kurichi kulam.geojson",
    "Perur": "Perur lake.geojson",
    "Singanallur": "Singanallur lake.geojson",
}

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:latest")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
RAG_DOCS_DIR = os.environ.get("RAG_DOCS_DIR", os.path.join(BASE_DIR, "rag_docs"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

RAG_VECTORSTORE = None
RAG_READY = False
RAG_LAST_ERROR = None
RAG_DOCUMENT_COUNT = 0
RAG_CHUNK_COUNT = 0


def load_rag_documents():
    """Load local knowledge documents for RAG."""
    if not os.path.exists(RAG_DOCS_DIR):
        return []

    documents = []
    supported_extensions = {".txt", ".md", ".pdf"}

    for root, _, files in os.walk(RAG_DOCS_DIR):
        for file_name in files:
            extension = os.path.splitext(file_name)[1].lower()
            if extension not in supported_extensions:
                continue

            file_path = os.path.join(root, file_name)
            try:
                if extension == ".pdf":
                    pdf_documents = PyPDFLoader(file_path).load()
                    for pdf_document in pdf_documents:
                        if pdf_document.page_content.strip():
                            pdf_document.metadata.update({"source": file_name, "path": file_path})
                            documents.append(pdf_document)
                else:
                    with open(file_path, "r", encoding="utf-8") as file_handle:
                        content = file_handle.read().strip()
                    if not content:
                        continue

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": file_name, "path": file_path}
                        )
                    )
            except Exception as error:
                print(f"RAG document read error for {file_path}: {error}")

    return documents


def initialize_rag_index():
    """Initialize LangChain RAG index using local documents and Ollama embeddings."""
    global RAG_VECTORSTORE, RAG_READY, RAG_LAST_ERROR, RAG_DOCUMENT_COUNT, RAG_CHUNK_COUNT

    if not LANGCHAIN_AVAILABLE:
        RAG_VECTORSTORE = None
        RAG_READY = False
        RAG_DOCUMENT_COUNT = 0
        RAG_CHUNK_COUNT = 0
        RAG_LAST_ERROR = f"LangChain imports unavailable: {LANGCHAIN_IMPORT_ERROR}"
        print(RAG_LAST_ERROR)
        return

    try:
        documents = load_rag_documents()
        if not documents:
            RAG_VECTORSTORE = None
            RAG_READY = False
            RAG_DOCUMENT_COUNT = 0
            RAG_CHUNK_COUNT = 0
            RAG_LAST_ERROR = f"No RAG documents found in {RAG_DOCS_DIR}"
            print(RAG_LAST_ERROR)
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120
        )
        chunks = splitter.split_documents(documents)
        RAG_DOCUMENT_COUNT = len(documents)
        RAG_CHUNK_COUNT = len(chunks)

        embedding_model = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        RAG_VECTORSTORE = FAISS.from_documents(chunks, embedding_model)
        RAG_READY = True
        RAG_LAST_ERROR = None
        print(f"RAG index initialized with {len(chunks)} chunks from {len(documents)} documents")
    except Exception as error:
        RAG_VECTORSTORE = None
        RAG_READY = False
        RAG_DOCUMENT_COUNT = 0
        RAG_CHUNK_COUNT = 0
        RAG_LAST_ERROR = str(error)
        print(f"RAG initialization failed: {error}")


def retrieve_rag_context(question, payload):
    """Retrieve top relevant document chunks for the current user query."""
    if not RAG_READY or RAG_VECTORSTORE is None:
        return []

    lake = payload.get("lake", {})
    lake_name = lake.get("name") or payload.get("lake_name") or "Unknown"
    retrieval_query = (
        f"Lake monitoring guidance for {lake_name}. "
        f"Question: {question}. "
        f"Water health: {lake.get('waterHealth', 'Unknown')}. "
        f"BOD: {lake.get('bodLevel', 'Unknown')}. "
        f"Pollution causes: {lake.get('pollutionCauses', 'Unknown')}."
    )

    try:
        docs = RAG_VECTORSTORE.similarity_search(retrieval_query, k=RAG_TOP_K)
        return docs
    except Exception as error:
        print(f"RAG retrieval error: {error}")
        return []


def load_lake_geometry(lake_name):
    """Load a lake geometry from the GeoJSON files directory."""
    filename = LAKE_GEOJSON_FILES.get(lake_name)
    if not filename:
        return None

    file_path = os.path.join(GEOJSON_DIR, filename)
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as file_handle:
            return json.load(file_handle)
    except Exception as error:
        print(f"Error loading geometry for {lake_name}: {error}")
        return None


def build_mock_lake(
    lake_id,
    lake_name,
    ndwi,
    ndci,
    fai,
    mci,
    swir_ratio,
    turbidity,
    bod_level,
    water_health,
    pollution_causes,
    suggestions,
    year,
):
    geometry = load_lake_geometry(lake_name)
    if geometry is None:
        geometry = {
            "type": "FeatureCollection",
            "features": [],
        }

    return {
        'id': lake_id,
        'name': lake_name,
        'ndwi': ndwi,
        'ndci': ndci,
        'fai': fai,
        'mci': mci,
        'swir_ratio': swir_ratio,
        'turbidity': turbidity,
        'bodLevel': bod_level,
        'waterHealth': water_health,
        'pollutionCauses': pollution_causes,
        'suggestions': suggestions,
        'geometry': geometry,
        'year': year,
    }

# Initialize Google Earth Engine
def initialize_earth_engine():
    """Initialize Google Earth Engine with proper authentication"""
    try:
        # Try to initialize with service account
        ee.Initialize(project='neer2025')
        print("Google Earth Engine initialized successfully with existing credentials")
        return True
    except Exception as e:
        print(f"Standard initialization failed: {e}")
        try:
            # Try to authenticate interactively without gcloud dependency
            try:
                ee.Authenticate(auth_mode='localhost')
            except Exception:
                ee.Authenticate(auth_mode='notebook')
            ee.Initialize(project='neer2025')
            print("Google Earth Engine initialized successfully after authentication")
            return True
        except Exception as auth_error:
            print(f"Earth Engine authentication failed: {auth_error}")
            print("Will use mock data as fallback")
            return False

# Global variable to track EE status
EE_INITIALIZED = initialize_earth_engine()

# Global initialization for LangChain RAG
initialize_rag_index()

@app.route('/')
def home():
    """Simple test route"""
    return jsonify({"message": "NEER Dashboard API is running!", "status": "success"})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/test')
def test_endpoint():
    """Test endpoint without Earth Engine"""
    return jsonify({
        "message": "API is working",
        "lakes_available": ["Ukkadam", "Valankulam", "Kurichi", "Perur", "Singanallur"],
        "status": "success"
    })

@app.route('/api/lakes/mock', methods=['GET'])
def get_mock_lakes():
    """Get mock lake data for testing"""
    year = request.args.get('year', 2024, type=int)
    
    mock_lakes = [
        build_mock_lake(
            'ukkadam',
            'Ukkadam',
            0.3,
            -0.1,
            0.02,
            15.2,
            1.2,
            850.5,
            15.45,
            'Poor',
            'High sediment, algal bloom',
            'Reduce catchment erosion, limit nutrient runoff',
            year,
        ),
        build_mock_lake(
            'valankulam',
            'Valankulam',
            0.5,
            -0.05,
            0.01,
            8.1,
            1.0,
            420.3,
            6.2,
            'Moderate',
            'Minor algal growth',
            'Monitor nutrient levels',
            year,
        ),
        build_mock_lake(
            'kurichi',
            'Kurichi',
            0.7,
            -0.15,
            0.005,
            5.3,
            0.8,
            180.1,
            3.8,
            'Good',
            'No major issues',
            'Continue current management',
            year,
        )
    ]
    
    return jsonify(mock_lakes)

def load_lakes_from_files():
    """Load all lake geometries from GeoJSON files"""
    lakes = {}
    
    # Ukkadam geometry (hardcoded from your original code)
    ukkadam_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Ukkadam"},
            "geometry": {
                "coordinates": [[[76.96095638648234, 10.988575303133587], 
                               [76.96051032283168, 10.988737482780266],
                               [76.95813131669479, 10.9880238916666], 
                               [76.95646270822363, 10.987018373985165],
                               [76.9540010977073, 10.985866889852815], 
                               [76.9518864255856, 10.985218164418924],
                               [76.9500856501075, 10.984958673846393], 
                               [76.94997000397598, 10.983920709273917],
                               [76.94919352280618, 10.983223324777526], 
                               [76.94821879112538, 10.982947614174023],
                               [76.94826835375284, 10.98236375557407], 
                               [76.94702928805697, 10.981925860866397],
                               [76.94699624630488, 10.980417551902818], 
                               [76.9468640792968, 10.9802391492576],
                               [76.94574065973245, 10.980222930829356], 
                               [76.94504678294214, 10.980093183376937],
                               [76.94524503345434, 10.979120075661697], 
                               [76.94600499374752, 10.9791038571732],
                               [76.9466327870337, 10.978730831677623], 
                               [76.94874745915558, 10.978925453734405],
                               [76.94957350295277, 10.979249823542816], 
                               [76.95020129623902, 10.979055201699836],
                               [76.95033346324595, 10.97866595763135], 
                               [76.95153948719098, 10.97843889835319],
                               [76.95226640573219, 10.978649739116705], 
                               [76.95350547142806, 10.978682176143948],
                               [76.95411674383888, 10.97847133540364], 
                               [76.95580187318546, 10.978422679827219],
                               [76.95813131669479, 10.977952342171605], 
                               [76.95958515377828, 10.978714613167696],
                               [76.96034511407152, 10.979266042023383], 
                               [76.96084074034962, 10.97997965430892],
                               [76.96128680400022, 10.983077360372121], 
                               [76.96130332487684, 10.984974892013383],
                               [76.96117115786876, 10.985493872901799], 
                               [76.9612702831248, 10.986029070987911],
                               [76.96095638648234, 10.988575303133587]]],
                "type": "Polygon"
            }
        }]
    }
    
    lakes["Ukkadam"] = ee.FeatureCollection(ukkadam_geojson)
    
    # Load other lakes from files
    lake_files = {
        "Valankulam": "geojson_files/valankulam(includes chinna kulam).geojson",
        "Kurichi": "geojson_files/Kurichi kulam.geojson",
        "Perur": "geojson_files/Perur lake.geojson",
        "Singanallur": "geojson_files/Singanallur lake.geojson"
    }
    
    for lake_name, filename in lake_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    geojson_data = json.load(f)
                    lakes[lake_name] = ee.FeatureCollection(geojson_data)
            except Exception as e:
                print(f"Error loading {lake_name}: {str(e)}")
                continue
    
    return lakes

def compute_indices(image):
    """Compute all water quality indices"""
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndci = image.normalizedDifference(['B5', 'B4']).rename('NDCI')
    fai = image.expression(
        '(B8 - B4) / (B8 + B4)',
        {'B8': image.select('B8'), 'B4': image.select('B4')}
    ).rename('FAI')
    mci = image.expression(
        'B5 - B4 - (B6 - B4) * ((705 - 665) / (740 - 665))',
        {'B5': image.select('B5'), 'B4': image.select('B4'), 'B6': image.select('B6')}
    ).rename('MCI')
    turbidity = image.select(['B2', 'B3', 'B4']).reduce(ee.Reducer.mean()).rename('Turbidity')
    swir_ratio = image.select('B11').divide(image.select('B12')).rename('SWIR_Ratio')
    return image.addBands([ndwi, ndci, fai, mci, turbidity, swir_ratio])

def classify_pollution(values):
    """Classify pollution causes and generate suggestions"""
    reasons = []
    suggestions = []

    if values.get('FAI', 0) > 0.05:
        reasons.append("Algal bloom")
        suggestions.append("Limit nutrient runoff")

    if values.get('NDWI', 0) < 0.2:
        reasons.append("Water scarcity")
        suggestions.append("Increase water inflow")

    if values.get('SWIR_Ratio', 0) > 1.5:
        reasons.append("Chemical or sediment pollution")
        suggestions.append("Investigate industrial discharges")

    if values.get('Turbidity', 0) > 1000:
        reasons.append("High sediment or garbage dumping")
        suggestions.append("Reduce catchment erosion / waste dumping")

    return ", ".join(reasons) or "No major issues", ", ".join(set(suggestions)) or "No action needed"


def build_ollama_prompt(payload):
    """Build a grounded prompt for local LLM recommendations."""
    return f"""
You are a water quality assistant for lake restoration planning.
Generate concise, practical recommendations based only on this lake data.

Lake: {payload.get('name', 'Unknown')}
Year: {payload.get('year', 'Unknown')}
Water Health: {payload.get('waterHealth', 'Unknown')}
NDWI: {payload.get('ndwi', 'Unknown')}
NDCI: {payload.get('ndci', 'Unknown')}
FAI: {payload.get('fai', 'Unknown')}
MCI: {payload.get('mci', 'Unknown')}
SWIR Ratio: {payload.get('swir_ratio', 'Unknown')}
Turbidity: {payload.get('turbidity', 'Unknown')}
BOD Level (mg/L): {payload.get('bodLevel', 'Unknown')}
Pollution Causes: {payload.get('pollutionCauses', 'Unknown')}

Return exactly:
1) a one-line summary,
2) three short actionable recommendations,
3) one caution note.
Keep total response under 80 words.
""".strip()


def get_ollama_suggestion(payload):
    """Generate a recommendation via local Ollama model."""
    api_url = f"{OLLAMA_BASE_URL}/api/chat"
    request_payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You produce grounded, concise environmental recommendations from provided numeric inputs."
            },
            {
                "role": "user",
                "content": build_ollama_prompt(payload)
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            response_json = json.loads(response.read().decode("utf-8"))
            return response_json.get("message", {}).get("content", "").strip()
    except urllib.error.URLError as error:
        print(f"Ollama connection error: {error}")
        return None
    except Exception as error:
        print(f"Ollama generation error: {error}")
        return None


def build_chat_context_prompt(payload):
    """Build a compact lake context prompt for chatbot conversations."""
    lake = payload.get("lake", {})
    lake_name = lake.get("name") or payload.get("lake_name") or "Unknown"
    return f"""
You are a helpful assistant for a lake monitoring dashboard.
Answer with concise, practical guidance using the provided lake context.

Lake: {lake_name}
Year: {payload.get('year', 'Unknown')}
Water Health: {lake.get('waterHealth', 'Unknown')}
NDWI: {lake.get('ndwi', 'Unknown')}
NDCI: {lake.get('ndci', 'Unknown')}
FAI: {lake.get('fai', 'Unknown')}
MCI: {lake.get('mci', 'Unknown')}
SWIR Ratio: {lake.get('swir_ratio', 'Unknown')}
Turbidity: {lake.get('turbidity', 'Unknown')}
BOD Level (mg/L): {lake.get('bodLevel', 'Unknown')}
Pollution Causes: {lake.get('pollutionCauses', 'Unknown')}
Existing Suggestions: {lake.get('suggestions', 'Unknown')}

Rules:
- Keep answers under 120 words.
- Use bullet points if helpful.
- Do not mention unsupported facts.
- If the question is outside the lake context, say so briefly and stay helpful.
""".strip()


def get_ollama_chat_response(payload):
    """Generate a chat reply via local Ollama model."""
    api_url = f"{OLLAMA_BASE_URL}/api/chat"
    history = payload.get("history", [])
    user_message = payload.get("message", "")
    retrieved_docs = retrieve_rag_context(user_message, payload)
    context_blocks = []
    for index, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(f"[{index}] Source: {source}\n{doc.page_content}")

    rag_context = "\n\n".join(context_blocks) if context_blocks else "No retrieved context available."

    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant for a water-quality dashboard. Ground your responses in the provided context."
        },
        {
            "role": "system",
            "content": build_chat_context_prompt(payload)
        },
        {
            "role": "system",
            "content": (
                "Retrieved knowledge context (RAG):\n"
                f"{rag_context}\n\n"
                "When possible, cite source numbers like [1], [2]."
            )
        },
    ]

    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    request_payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            response_json = json.loads(response.read().decode("utf-8"))
            return {
                "reply": response_json.get("message", {}).get("content", "").strip(),
                "retrieved_count": len(retrieved_docs)
            }
    except urllib.error.URLError as error:
        print(f"Ollama chat connection error: {error}")
        return None
    except Exception as error:
        print(f"Ollama chat generation error: {error}")
        return None


@app.route('/api/ai-suggestions', methods=['POST'])
def get_ai_suggestions():
    """Generate local-model lake recommendations using Ollama."""
    payload = request.get_json(silent=True) or {}

    fallback_suggestion = payload.get('suggestions', 'No additional suggestion available')
    suggestion = get_ollama_suggestion(payload)

    if suggestion:
        return jsonify({
            'suggestion': suggestion,
            'source': 'ollama',
            'model': OLLAMA_MODEL
        })

    return jsonify({
        'suggestion': fallback_suggestion,
        'source': 'rule-based-fallback',
        'model': None
    })


@app.route('/api/chat', methods=['POST'])
def chat_assistant():
    """Chat with the local Ollama assistant using lake context."""
    payload = request.get_json(silent=True) or {}
    response = get_ollama_chat_response(payload)

    if response and response.get("reply"):
        return jsonify({
            'reply': response.get("reply"),
            'source': 'rag+ollama' if response.get("retrieved_count", 0) > 0 else 'ollama',
            'model': OLLAMA_MODEL,
            'retrieved_count': response.get("retrieved_count", 0),
            'rag_ready': RAG_READY,
        })

    lake = payload.get('lake', {})
    lake_name = lake.get('name') or payload.get('lake_name') or 'the selected lake'
    fallback_reply = (
        f"I can help with {lake_name}. Try asking about causes, trends, or actions. "
        f"Current health: {lake.get('waterHealth', 'Unknown')}."
    )
    return jsonify({
        'reply': fallback_reply,
        'source': 'fallback',
        'model': None
    })


@app.route('/api/rag/status', methods=['GET'])
def rag_status():
    """Get the status of LangChain RAG readiness."""
    return jsonify({
        'langchain_available': LANGCHAIN_AVAILABLE,
        'rag_ready': RAG_READY,
        'docs_dir': RAG_DOCS_DIR,
        'embedding_model': OLLAMA_EMBED_MODEL,
        'document_count': RAG_DOCUMENT_COUNT,
        'chunk_count': RAG_CHUNK_COUNT,
        'error': RAG_LAST_ERROR,
    })


@app.route('/api/rag/reindex', methods=['POST'])
def rag_reindex():
    """Rebuild the LangChain RAG index from local documents."""
    initialize_rag_index()
    return jsonify({
        'langchain_available': LANGCHAIN_AVAILABLE,
        'rag_ready': RAG_READY,
        'document_count': RAG_DOCUMENT_COUNT,
        'chunk_count': RAG_CHUNK_COUNT,
        'error': RAG_LAST_ERROR,
    })

def get_mock_lakes_response(year):
    """Return mock data response for testing when Earth Engine is not available"""
    mock_lakes = [
        build_mock_lake(
            'ukkadam',
            'Ukkadam',
            0.3,
            -0.1,
            0.02,
            15.2,
            1.2,
            850.5,
            15.45,
            'Poor',
            'High sediment, algal bloom',
            'Reduce catchment erosion, limit nutrient runoff',
            year,
        ),
        build_mock_lake(
            'valankulam',
            'Valankulam',
            0.5,
            -0.05,
            0.01,
            8.1,
            1.0,
            420.3,
            6.2,
            'Moderate',
            'Minor algal growth',
            'Monitor nutrient levels',
            year,
        ),
        build_mock_lake(
            'kurichi',
            'Kurichi',
            0.7,
            -0.15,
            0.005,
            5.3,
            0.8,
            180.1,
            3.8,
            'Good',
            'No major issues',
            'Continue current management',
            year,
        ),
        build_mock_lake(
            'perur',
            'Perur',
            0.4,
            -0.08,
            0.015,
            12.5,
            1.1,
            650.2,
            8.1,
            'Poor',
            'Chemical pollution, sediment',
            'Investigate industrial discharges, reduce erosion',
            year,
        ),
        build_mock_lake(
            'singanallur',
            'Singanallur',
            0.6,
            -0.12,
            0.008,
            6.8,
            0.9,
            320.4,
            5.3,
            'Moderate',
            'Moderate nutrient loading',
            'Control agricultural runoff',
            year,
        )
    ]
    
    return jsonify(mock_lakes)

@app.route('/api/lakes', methods=['GET'])
def get_all_lakes():
    """Get all lakes with current water quality data"""
    year = request.args.get('year', 2024, type=int)
    
    # Validate year
    if year < 2015 or year > 2025:
        return jsonify({'error': 'Invalid year. Please use years between 2015-2025'}), 400
    
    try:
        print(f"Attempting to get real data for year {year}")
        lakes = load_lakes_from_files()
        print(f"Loaded {len(lakes)} lakes from files")
        
        # Try to get Sentinel-2 data
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterDate(start, end) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B11', 'B12']) \
            .median()

        s2 = compute_indices(s2)

        results = []

        for lake_name, lake_fc in lakes.items():
            try:
                print(f"Processing lake: {lake_name}")
                stats = s2.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=lake_fc.geometry(),
                    scale=10,
                    maxPixels=1e9
                ).getInfo()

                if stats and 'NDWI' in stats and stats['NDWI'] is not None:
                    bod = 26.303 * stats['NDWI'] + 7.546

                    if bod > 8:
                        health = "Poor"
                    elif bod > 4:
                        health = "Moderate"
                    else:
                        health = "Good"

                    reasons, suggestions = classify_pollution(stats)
                    geometry = lake_fc.getInfo()

                    results.append({
                        'id': lake_name.lower().replace(' ', '_'),
                        'name': lake_name,
                        'ndwi': round(stats.get('NDWI', 0), 4),
                        'ndci': round(stats.get('NDCI', 0), 4),
                        'fai': round(stats.get('FAI', 0), 4),
                        'mci': round(stats.get('MCI', 0), 4),
                        'swir_ratio': round(stats.get('SWIR_Ratio', 0), 4),
                        'turbidity': round(stats.get('Turbidity', 0), 2),
                        'bodLevel': round(bod, 2),
                        'waterHealth': health,
                        'pollutionCauses': reasons,
                        'suggestions': suggestions,
                        'geometry': geometry,
                        'year': year
                    })
                    print(f"Successfully processed {lake_name}")
                else:
                    print(f"No valid stats for {lake_name}")
            except Exception as error:
                print(f"Error processing lake {lake_name}: {error}")
                continue

        if results:
            print(f"Returning {len(results)} real lake results")
            return jsonify(results)

        print("No real data available, falling back to mock data")
        return get_mock_lakes_response(year)

    except Exception as e:
        print(f"Earth Engine error: {str(e)}")
        print("Falling back to mock data due to Earth Engine issues")
        return get_mock_lakes_response(year)

@app.route('/api/lakes/<lake_id>/history', methods=['GET'])
def get_lake_history(lake_id):
    """Get historical data for a specific lake with trend analysis"""
    start_year = request.args.get('start_year', 2020, type=int)
    end_year = request.args.get('end_year', 2024, type=int)
    
    # Validate year range
    if start_year > end_year or start_year < 2015 or end_year > 2025:
        return jsonify({'error': 'Invalid year range. Please use years between 2015-2025'}), 400
    
    try:
        # Map lake_id to actual lake names
        lake_mapping = {
            'ukkadam': 'Ukkadam',
            'valankulam': 'Valankulam',
            'kurichi': 'Kurichi',
            'perur': 'Perur',
            'singanallur': 'Singanallur'
        }
        
        lake_name = lake_mapping.get(lake_id.lower())
        
        if not lake_name:
            return jsonify({'error': 'Lake not found'}), 404

        if EE_INITIALIZED:
            return get_real_historical_data(lake_name, start_year, end_year)
        else:
            return get_mock_historical_data(lake_id, start_year, end_year)
        
    except Exception as e:
        print(f"Error in get_lake_history: {str(e)}")
        return get_mock_historical_data(lake_id, start_year, end_year)

def get_real_historical_data(lake_name, start_year, end_year):
    """Get real historical data from Earth Engine"""
    try:
        lakes = load_lakes_from_files()
        
        if lake_name not in lakes:
            return jsonify({'error': 'Lake not found'}), 404
        
        lake_fc = lakes[lake_name]
        historical_data = []
        trend_analysis = {"improving": 0, "degrading": 0, "stable": 0}
        
        previous_bod = None
        
        for year in range(start_year, end_year + 1):
            try:
                start = f"{year}-01-01"
                end_date = f"{year}-12-31"
                s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
                    .filterDate(start, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
                    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B11', 'B12']) \
                    .median()
                
                s2 = compute_indices(s2)
                
                stats = s2.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=lake_fc.geometry(),
                    scale=10,
                    maxPixels=1e9
                ).getInfo()
                
                if stats and 'NDWI' in stats and stats['NDWI'] is not None:
                    bod = 26.303 * stats['NDWI'] + 7.546
                    health = "Poor" if bod > 8 else "Moderate" if bod > 4 else "Good"
                    
                    # Trend analysis
                    if previous_bod is not None:
                        if bod < previous_bod - 1:
                            trend = "improving"
                            trend_analysis["improving"] += 1
                        elif bod > previous_bod + 1:
                            trend = "degrading"
                            trend_analysis["degrading"] += 1
                        else:
                            trend = "stable"
                            trend_analysis["stable"] += 1
                    else:
                        trend = "baseline"
                    
                    historical_data.append({
                        'year': year,
                        'ndwi': round(stats['NDWI'], 4),
                        'ndci': round(stats.get('NDCI', 0), 4),
                        'fai': round(stats.get('FAI', 0), 4),
                        'mci': round(stats.get('MCI', 0), 4),
                        'bodLevel': round(bod, 2),
                        'waterHealth': health,
                        'trend': trend,
                        'turbidity': round(stats.get('Turbidity', 0), 2),
                        'swir_ratio': round(stats.get('SWIR_Ratio', 0), 4)
                    })
                    
                    previous_bod = bod
                    
            except Exception as e:
                print(f"Error processing year {year}: {str(e)}")
                continue
        
        # Calculate overall trend
        if trend_analysis["degrading"] > trend_analysis["improving"]:
            overall_trend = "degrading"
        elif trend_analysis["improving"] > trend_analysis["degrading"]:
            overall_trend = "improving"
        else:
            overall_trend = "stable"
        
        return jsonify({
            'historical_data': historical_data,
            'trend_analysis': {
                'overall_trend': overall_trend,
                'trend_counts': trend_analysis,
                'data_points': len(historical_data)
            }
        })
        
    except Exception as e:
        print(f"Error in get_real_historical_data: {str(e)}")
        return get_mock_historical_data(lake_name.lower(), start_year, end_year)

def get_mock_historical_data(lake_id, start_year, end_year):
    """Generate mock historical data with realistic trends"""
    import random
    
    # Base values for different lakes
    base_values = {
        'ukkadam': {'base_ndwi': 0.3, 'trend': -0.02},  # Degrading
        'valankulam': {'base_ndwi': 0.5, 'trend': 0.01},  # Improving
        'kurichi': {'base_ndwi': 0.7, 'trend': 0.005},   # Stable/Improving
        'perur': {'base_ndwi': 0.4, 'trend': -0.015},    # Degrading
        'singanallur': {'base_ndwi': 0.6, 'trend': 0.008} # Improving
    }
    
    lake_config = base_values.get(lake_id, {'base_ndwi': 0.5, 'trend': 0})
    historical_data = []
    trend_analysis = {"improving": 0, "degrading": 0, "stable": 0}
    previous_bod = None
    
    for i, year in enumerate(range(start_year, end_year + 1)):
        # Calculate NDWI with trend and some random variation
        ndwi = lake_config['base_ndwi'] + (lake_config['trend'] * i) + random.uniform(-0.05, 0.05)
        ndwi = max(0, min(1, ndwi))  # Clamp between 0 and 1
        
        bod = 26.303 * ndwi + 7.546
        health = "Poor" if bod > 8 else "Moderate" if bod > 4 else "Good"
        
        # Trend analysis
        if previous_bod is not None:
            if bod < previous_bod - 1:
                trend = "improving"
                trend_analysis["improving"] += 1
            elif bod > previous_bod + 1:
                trend = "degrading"
                trend_analysis["degrading"] += 1
            else:
                trend = "stable"
                trend_analysis["stable"] += 1
        else:
            trend = "baseline"
        
        historical_data.append({
            'year': year,
            'ndwi': round(ndwi, 4),
            'ndci': round(random.uniform(-0.2, 0.1), 4),
            'fai': round(random.uniform(0, 0.05), 4),
            'mci': round(random.uniform(5, 20), 2),
            'bodLevel': round(bod, 2),
            'waterHealth': health,
            'trend': trend,
            'turbidity': round(random.uniform(100, 1000), 2),
            'swir_ratio': round(random.uniform(0.8, 1.5), 4)
        })
        
        previous_bod = bod
    
    # Calculate overall trend
    if trend_analysis["degrading"] > trend_analysis["improving"]:
        overall_trend = "degrading"
    elif trend_analysis["improving"] > trend_analysis["degrading"]:
        overall_trend = "improving"
    else:
        overall_trend = "stable"
    
    return jsonify({
        'historical_data': historical_data,
        'trend_analysis': {
            'overall_trend': overall_trend,
            'trend_counts': trend_analysis,
            'data_points': len(historical_data)
        }
    })

@app.route('/api/alerts', methods=['GET'])
def get_water_quality_alerts():
    """Get water quality alerts for rapidly degrading lakes"""
    try:
        # Check if Earth Engine is available (similar check as other endpoints)
        try:
            ee.Number(1).getInfo()
            return get_real_alerts()
        except:
            return get_mock_alerts()
    except Exception as e:
        print(f"Error in get_water_quality_alerts: {str(e)}")
        return get_mock_alerts()

def get_real_alerts():
    """Get real alerts based on Earth Engine data"""
    try:
        lakes = load_lakes_from_files()
        alerts = []
        
        current_year = 2024
        
        for lake_name, lake_fc in lakes.items():
            try:
                # Get recent data (last 2 years)
                start = f"{current_year-1}-01-01"
                end_date = f"{current_year}-12-31"
                
                s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
                    .filterDate(start, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
                    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B11', 'B12'])
                
                # Get data for last year and current year
                last_year = s2.filterDate(f"{current_year-1}-01-01", f"{current_year-1}-12-31").median()
                current_year_data = s2.filterDate(f"{current_year}-01-01", f"{current_year}-12-31").median()
                
                last_year = compute_indices(last_year)
                current_year_data = compute_indices(current_year_data)
                
                # Get statistics
                last_stats = last_year.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=lake_fc.geometry(),
                    scale=10,
                    maxPixels=1e9
                ).getInfo()
                
                current_stats = current_year_data.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=lake_fc.geometry(),
                    scale=10,
                    maxPixels=1e9
                ).getInfo()
                
                if (last_stats and current_stats and 
                    'NDWI' in last_stats and 'NDWI' in current_stats and
                    last_stats['NDWI'] is not None and current_stats['NDWI'] is not None):
                    
                    last_bod = 26.303 * last_stats['NDWI'] + 7.546
                    current_bod = 26.303 * current_stats['NDWI'] + 7.546
                    
                    bod_change = current_bod - last_bod
                    
                    # Generate alerts based on various criteria
                    if bod_change > 3:  # Significant increase in BOD
                        alerts.append({
                            'id': f"alert_{lake_name}_{current_year}",
                            'lake_name': lake_name.title(),
                            'alert_type': 'degrading_water_quality',
                            'severity': 'high' if bod_change > 5 else 'medium',
                            'message': f"Water quality rapidly degrading. BOD increased by {bod_change:.1f} mg/L",
                            'timestamp': f"{current_year}-12-01T00:00:00Z",
                            'current_bod': round(current_bod, 2),
                            'previous_bod': round(last_bod, 2),
                            'change': round(bod_change, 2),
                            'recommended_action': 'Immediate investigation and pollution source assessment required'
                        })
                    
                    # Additional pollution indicators
                    if current_stats.get('NDCI', 0) > 0.2:  # High algae
                        alerts.append({
                            'id': f"algae_{lake_name}_{current_year}",
                            'lake_name': lake_name.title(),
                            'alert_type': 'algal_bloom',
                            'severity': 'medium',
                            'message': f"Potential algal bloom detected (NDCI: {current_stats['NDCI']:.3f})",
                            'timestamp': f"{current_year}-11-15T00:00:00Z",
                            'recommended_action': 'Monitor nutrient levels and implement algae control measures'
                        })
                    
                    if current_stats.get('Turbidity', 0) > 800:  # High turbidity
                        alerts.append({
                            'id': f"turbidity_{lake_name}_{current_year}",
                            'lake_name': lake_name.title(),
                            'alert_type': 'high_turbidity',
                            'severity': 'medium',
                            'message': f"High turbidity detected ({current_stats['Turbidity']:.1f} NTU)",
                            'timestamp': f"{current_year}-11-20T00:00:00Z",
                            'recommended_action': 'Check for erosion sources and sediment runoff'
                        })
                        
            except Exception as e:
                print(f"Error processing alerts for {lake_name}: {str(e)}")
                continue
        
        return jsonify({
            'alerts': alerts,
            'total_alerts': len(alerts),
            'last_updated': f"{current_year}-12-01T00:00:00Z"
        })
        
    except Exception as e:
        print(f"Error in get_real_alerts: {str(e)}")
        return get_mock_alerts()

def get_mock_alerts():
    """Generate mock alerts for demonstration"""
    alerts = [
        {
            'id': 'alert_ukkadam_2024',
            'lake_name': 'Ukkadam Lake',
            'alert_type': 'degrading_water_quality',
            'severity': 'high',
            'message': 'Water quality rapidly degrading. BOD increased by 4.2 mg/L in past year',
            'timestamp': '2024-11-25T08:30:00Z',
            'current_bod': 12.5,
            'previous_bod': 8.3,
            'change': 4.2,
            'recommended_action': 'Immediate investigation and pollution source assessment required'
        },
        {
            'id': 'algae_perur_2024',
            'lake_name': 'Perur Lake',
            'alert_type': 'algal_bloom',
            'severity': 'medium',
            'message': 'Potential algal bloom detected (NDCI: 0.245)',
            'timestamp': '2024-11-20T14:15:00Z',
            'recommended_action': 'Monitor nutrient levels and implement algae control measures'
        },
        {
            'id': 'turbidity_singanallur_2024',
            'lake_name': 'Singanallur Lake',
            'alert_type': 'high_turbidity',
            'severity': 'medium',
            'message': 'High turbidity detected (850.3 NTU)',
            'timestamp': '2024-11-18T11:45:00Z',
            'recommended_action': 'Check for erosion sources and sediment runoff'
        },
        {
            'id': 'pollution_valankulam_2024',
            'lake_name': 'Valankulam',
            'alert_type': 'pollution_source',
            'severity': 'high',
            'message': 'New pollution source detected in northeast catchment area',
            'timestamp': '2024-11-15T16:20:00Z',
            'recommended_action': 'Investigate industrial discharge and implement immediate containment'
        }
    ]
    
    return jsonify({
        'alerts': alerts,
        'total_alerts': len(alerts),
        'last_updated': '2024-11-25T12:00:00Z'
    })

@app.route('/api/pollution-sources/<lake_id>', methods=['GET'])
def get_pollution_sources(lake_id):
    """Get detailed pollution source mapping for a specific lake"""
    try:
        # Check if Earth Engine is available
        try:
            ee.Number(1).getInfo()
            return get_real_pollution_sources(lake_id)
        except:
            return get_mock_pollution_sources(lake_id)
    except Exception as e:
        print(f"Error in get_pollution_sources: {str(e)}")
        return get_mock_pollution_sources(lake_id)

def get_real_pollution_sources(lake_id):
    """Get real pollution sources using Earth Engine land use analysis"""
    try:
        lakes = load_lakes_from_files()
        
        lake_name = lake_id.replace('_', ' ').title()
        matching_lakes = [name for name in lakes.keys() if lake_name.lower() in name.lower()]
        
        if not matching_lakes:
            return jsonify({'error': 'Lake not found'}), 404
        
        lake_name = matching_lakes[0]
        lake_fc = lakes[lake_name]
        
        # Create buffer around lake for catchment analysis
        catchment = lake_fc.geometry().buffer(2000)  # 2km buffer
        
        # Get land use data (using Sentinel-2 for basic classification)
        s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterDate('2023-01-01', '2024-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']) \
            .median()
        
        # Simple land use classification
        ndvi = s2.normalizedDifference(['B8', 'B4'])
        ndbi = s2.normalizedDifference(['B11', 'B8'])
        mndwi = s2.normalizedDifference(['B3', 'B11'])
        
        # Classify land use
        urban = ndbi.gt(0.1).And(ndvi.lt(0.2))
        industrial = ndbi.gt(0.2).And(ndvi.lt(0.1))
        water = mndwi.gt(0.3)
        vegetation = ndvi.gt(0.4)
        
        # Calculate areas
        urban_area = urban.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=catchment,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        industrial_area = industrial.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=catchment,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        water_area = water.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=catchment,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        vegetation_area = vegetation.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=catchment,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        total_area = catchment.area().getInfo()
        
        # Calculate pollution risk scores
        urban_percent = (urban_area.get('nd', 0) / total_area) * 100
        industrial_percent = (industrial_area.get('nd', 0) / total_area) * 100
        
        pollution_risk = min(100, (urban_percent * 0.6) + (industrial_percent * 1.5))
        
        return jsonify({
            'lake_name': lake_name,
            'catchment_analysis': {
                'total_area_km2': round(total_area / 1000000, 2),
                'urban_coverage_percent': round(urban_percent, 1),
                'industrial_coverage_percent': round(industrial_percent, 1),
                'vegetation_coverage_percent': round((vegetation_area.get('nd', 0) / total_area) * 100, 1),
                'water_coverage_percent': round((water_area.get('nd', 0) / total_area) * 100, 1)
            },
            'pollution_risk_score': round(pollution_risk, 1),
            'risk_level': 'High' if pollution_risk > 70 else 'Medium' if pollution_risk > 40 else 'Low',
            'identified_sources': get_identified_sources(lake_id, pollution_risk),
            'recommendations': get_pollution_recommendations(pollution_risk, urban_percent, industrial_percent)
        })
        
    except Exception as e:
        print(f"Error in get_real_pollution_sources: {str(e)}")
        return get_mock_pollution_sources(lake_id)

def get_mock_pollution_sources(lake_id):
    """Generate mock pollution source mapping data"""
    
    pollution_data = {
        'ukkadam': {
            'risk_score': 85,
            'urban_percent': 45,
            'industrial_percent': 25,
            'sources': [
                {'type': 'Industrial Discharge', 'severity': 'High', 'distance_km': 0.8},
                {'type': 'Urban Runoff', 'severity': 'High', 'distance_km': 0.3},
                {'type': 'Sewage Treatment Plant', 'severity': 'Medium', 'distance_km': 1.2}
            ]
        },
        'valankulam': {
            'risk_score': 65,
            'urban_percent': 35,
            'industrial_percent': 15,
            'sources': [
                {'type': 'Agricultural Runoff', 'severity': 'Medium', 'distance_km': 1.5},
                {'type': 'Urban Runoff', 'severity': 'Medium', 'distance_km': 0.6},
                {'type': 'Construction Activities', 'severity': 'Low', 'distance_km': 2.1}
            ]
        },
        'kurichi': {
            'risk_score': 45,
            'urban_percent': 25,
            'industrial_percent': 8,
            'sources': [
                {'type': 'Residential Wastewater', 'severity': 'Medium', 'distance_km': 0.9},
                {'type': 'Road Runoff', 'severity': 'Low', 'distance_km': 0.4}
            ]
        },
        'perur': {
            'risk_score': 75,
            'urban_percent': 40,
            'industrial_percent': 20,
            'sources': [
                {'type': 'Textile Industry', 'severity': 'High', 'distance_km': 1.1},
                {'type': 'Urban Runoff', 'severity': 'Medium', 'distance_km': 0.5},
                {'type': 'Market Waste', 'severity': 'Medium', 'distance_km': 0.7}
            ]
        },
        'singanallur': {
            'risk_score': 55,
            'urban_percent': 30,
            'industrial_percent': 12,
            'sources': [
                {'type': 'Urban Runoff', 'severity': 'Medium', 'distance_km': 0.4},
                {'type': 'Small Industries', 'severity': 'Medium', 'distance_km': 1.3},
                {'type': 'Agricultural Runoff', 'severity': 'Low', 'distance_km': 2.0}
            ]
        }
    }
    
    lake_data = pollution_data.get(lake_id, pollution_data['ukkadam'])
    
    return jsonify({
        'lake_name': lake_id.replace('_', ' ').title(),
        'catchment_analysis': {
            'total_area_km2': 12.5,
            'urban_coverage_percent': lake_data['urban_percent'],
            'industrial_coverage_percent': lake_data['industrial_percent'],
            'vegetation_coverage_percent': max(0, 100 - lake_data['urban_percent'] - lake_data['industrial_percent'] - 10),
            'water_coverage_percent': 10
        },
        'pollution_risk_score': lake_data['risk_score'],
        'risk_level': 'High' if lake_data['risk_score'] > 70 else 'Medium' if lake_data['risk_score'] > 40 else 'Low',
        'identified_sources': lake_data['sources'],
        'recommendations': get_pollution_recommendations(
            lake_data['risk_score'], 
            lake_data['urban_percent'], 
            lake_data['industrial_percent']
        )
    })

def get_identified_sources(lake_id, pollution_risk):
    """Generate identified pollution sources based on analysis"""
    sources = []
    
    if pollution_risk > 70:
        sources.extend([
            {'type': 'Industrial Discharge', 'severity': 'High', 'distance_km': 1.2},
            {'type': 'Urban Runoff', 'severity': 'High', 'distance_km': 0.5}
        ])
    elif pollution_risk > 40:
        sources.extend([
            {'type': 'Urban Runoff', 'severity': 'Medium', 'distance_km': 0.8},
            {'type': 'Agricultural Runoff', 'severity': 'Medium', 'distance_km': 1.5}
        ])
    else:
        sources.append({'type': 'Natural Runoff', 'severity': 'Low', 'distance_km': 2.0})
    
    return sources

def get_pollution_recommendations(risk_score, urban_percent, industrial_percent):
    """Generate recommendations based on pollution analysis"""
    recommendations = []
    
    if risk_score > 70:
        recommendations.extend([
            "Immediate enforcement of industrial discharge regulations",
            "Install real-time water quality monitoring systems",
            "Implement emergency pollution response protocols"
        ])
    
    if urban_percent > 30:
        recommendations.append("Improve urban stormwater management and sewage treatment")
    
    if industrial_percent > 15:
        recommendations.append("Conduct detailed industrial effluent audits and implement stricter controls")
    
    recommendations.extend([
        "Establish buffer zones with native vegetation around the lake",
        "Regular community awareness programs on water conservation",
        "Quarterly water quality assessments and public reporting"
    ])
    
    return recommendations

if __name__ == '__main__':
    # Use environment variable for port (required for Railway/Heroku)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
