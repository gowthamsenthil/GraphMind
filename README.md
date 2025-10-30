# Agentic TensorFlow Debugger

AI-Powered TensorFlow Debugging with ReAct Reasoning & Self-Learning capabilities.

## Features

- **ReAct Reasoning Framework** 🧠 - Advanced reasoning with Thought-Action-Observation cycles
- **Knowledge Graph Integration** 🕸️ - Entity relationship mapping for better context
- **Self-Reflection** 🔄 - Quality assessment and confidence scoring
- **Continuous Learning** 📈 - System improves from user feedback
- **Agentic Updates** 🤖 - Intelligent knowledge base expansion

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**

```bash
cd /Users/gowtham/CascadeProjects/GraphMind/GraphMind
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Ensure artifacts are present:**

Make sure the following files exist in the project directory:
- `embeddings.npy`
- `faiss_index.index`
- `processed_docs.json`
- `kg_networkx.gpickle`

## Usage

### Starting the Server

```bash
python server.py
```

The server will start on `http://localhost:5000`

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Enter your TensorFlow error message in the input field
3. Click "Analyze Error" to get AI-powered debugging assistance
4. Review the solution with:
   - Root cause analysis
   - Step-by-step fixes
   - Code examples
   - Confidence scores
5. Provide feedback to help the system learn

### API Endpoints

#### Analyze Error
```bash
POST /api/analyze
Content-Type: application/json

{
  "error_message": "Your TensorFlow error here",
  "session_id": "optional-session-id"
}
```

#### Submit Feedback
```bash
POST /api/feedback
Content-Type: application/json

{
  "session_id": "session-id-from-analyze",
  "worked": true,
  "new_error": "Optional additional info if it didn't work"
}
```

#### Get Learning Statistics
```bash
GET /api/learning/stats
```

#### Apply Learning Updates
```bash
POST /api/learning/apply
```

#### Get System Statistics
```bash
GET /api/system/stats
```

## Configuration

### API Key

Update the `OPENROUTER_API_KEY` in `app_core.py` with your OpenRouter API key:

```python
OPENROUTER_API_KEY = "your-api-key-here"
```

Get your free API key from [OpenRouter](https://openrouter.ai/)

### Model Selection

Change the model in `app_core.py`:

```python
MODEL_NAME = "nvidia/nemotron-nano-9b-v2:free"  # or any other OpenRouter model
```

### Artifacts Directory

If your artifacts are in a different location, update:

```python
ARTIFACTS_DIR = "./path/to/artifacts"
```

## Architecture

### Core Components

1. **OpenRouterLLM** - API client for language model interactions
2. **ReasoningPlanner** - Plans reasoning strategy using ReAct framework
3. **EnhancedRetriever** - Multi-stage retrieval with knowledge graph awareness
4. **AdvancedReActAgent** - Main agent with reasoning and reflection
5. **FeedbackLearningSystem** - Handles user feedback and system updates

### Files Structure

```
GraphMind/
├── app_core.py          # Core logic (no UI dependencies)
├── server.py            # Flask server with API endpoints
├── templates/
│   └── index.html       # Web interface
├── static/
│   ├── css/
│   │   └── style.css    # Styling
│   └── js/
│       └── app.js       # Frontend JavaScript
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── embeddings.npy      # Document embeddings
├── faiss_index.index   # FAISS vector index
├── processed_docs.json # Knowledge base documents
└── kg_networkx.gpickle # Knowledge graph
```

## Learning System

The system continuously improves through:

1. **Feedback Collection** - Users report if solutions worked
2. **ReAct Analysis** - AI analyzes failures to extract insights
3. **Knowledge Base Updates** - New documents added from validated feedback
4. **Graph Enhancement** - Entity relationships strengthened
5. **Embedding Updates** - FAISS index expanded with new knowledge

## Troubleshooting

### Artifacts Not Found

Ensure all artifact files are in the correct directory:
```bash
ls -la *.npy *.index *.json *.gpickle
```

### Port Already in Use

Change the port in `server.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### API Errors

Check your OpenRouter API key is valid and has credits.

### Memory Issues

For large knowledge bases, increase Python memory:
```bash
export PYTHONMALLOC=malloc
```

## Development

### Running in Debug Mode

The server runs in debug mode by default. Disable for production:

```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

### Adding New Features

1. Core logic goes in `app_core.py`
2. API endpoints go in `server.py`
3. UI updates go in `templates/index.html` and `static/`

## License

MIT License

## Credits

- Powered by NVIDIA Nemotron via OpenRouter
- Built with Flask, FAISS, and NetworkX
- Uses Sentence Transformers for embeddings
