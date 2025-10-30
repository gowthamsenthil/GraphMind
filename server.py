from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import uuid

from app_core import (
    OpenRouterLLM,
    AdvancedReActAgent,
    FeedbackLearningSystem,
    load_all_artifacts,
    OPENROUTER_API_KEY,
    MODEL_NAME
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Global state
artifacts = None
llm = None
agent = None
learning_system = None
chat_sessions = {}

def initialize():
    """Initialize the application."""
    global artifacts, llm, agent, learning_system
    
    print("Loading artifacts...")
    artifacts = load_all_artifacts()
    
    if artifacts:
        llm = OpenRouterLLM(OPENROUTER_API_KEY, MODEL_NAME)
        agent = AdvancedReActAgent(artifacts, llm)
        learning_system = FeedbackLearningSystem(
            artifacts,
            artifacts["model"],
            llm
        )
        print("System initialized successfully!")
        return True
    else:
        print("Failed to load artifacts")
        return False

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze an error message."""
    try:
        data = request.json
        error_message = data.get('error_message', '')
        
        if not error_message:
            return jsonify({'error': 'No error message provided'}), 400
        
        # Perform analysis
        result = agent.analyze(error_message)
        
        # Store in session for feedback
        session_id = data.get('session_id') or str(uuid.uuid4())
        chat_sessions[session_id] = result
        result['session_id'] = session_id
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback on a solution."""
    try:
        data = request.json
        session_id = data.get('session_id')
        worked = data.get('worked', False)
        new_error = data.get('new_error', None)
        
        if not session_id or session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = chat_sessions[session_id]
        feedback = learning_system.collect_feedback(session_data, worked, new_error)
        
        return jsonify({
            'success': True,
            'feedback': feedback,
            'pending_updates': len(learning_system.update_queue)
        })
    
    except Exception as e:
        print(f"Error in submit_feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning/stats', methods=['GET'])
def get_learning_stats():
    """Get learning system statistics."""
    try:
        stats = learning_system.get_learning_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in get_learning_stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning/apply', methods=['POST'])
def apply_learning():
    """Apply pending learning updates."""
    try:
        result = learning_system.apply_learning_with_react()
        return jsonify(result)
    except Exception as e:
        print(f"Error in apply_learning: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning/feedback-log', methods=['GET'])
def get_feedback_log():
    """Get feedback log."""
    try:
        return jsonify({
            'feedback_log': learning_system.feedback_log,
            'update_queue': learning_system.update_queue
        })
    except Exception as e:
        print(f"Error in get_feedback_log: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics."""
    try:
        if not artifacts:
            return jsonify({'error': 'System not initialized'}), 500
        
        stats = {
            'total_docs': len(artifacts['docs']),
            'graph_nodes': artifacts['graph'].number_of_nodes(),
            'graph_edges': artifacts['graph'].number_of_edges(),
            'learned_docs': sum(1 for doc in artifacts['docs'] if doc.get('source') == 'feedback'),
            'model_name': MODEL_NAME
        }
        return jsonify(stats)
    except Exception as e:
        print(f"Error in get_system_stats: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if initialize():
        print("\n" + "="*60)
        print("🤖 Agentic TensorFlow Debugger - Flask Server")
        print("="*60)
        print("Server starting on http://localhost:5001")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to initialize. Please check artifacts directory.")