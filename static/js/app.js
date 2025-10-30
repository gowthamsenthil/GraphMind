// Global state
let currentSessionId = null;

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        
        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`${tab}-tab`).classList.add('active');
        
        // Load stats if learning tab
        if (tab === 'learning') {
            loadLearningStats();
            loadSystemStats();
        }
    });
});

// Analyze error
document.getElementById('analyze-btn').addEventListener('click', analyzeError);
document.getElementById('error-input').addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeError();
    }
});

async function analyzeError() {
    const input = document.getElementById('error-input');
    const btn = document.getElementById('analyze-btn');
    const errorMessage = input.value.trim();
    
    if (!errorMessage) {
        alert('Please enter an error message');
        return;
    }
    
    // Show loading
    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loading').style.display = 'inline-block';
    
    // Add user message
    addMessage('user', errorMessage);
    input.value = '';
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ error_message: errorMessage, session_id: currentSessionId })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentSessionId = result.session_id;
            addAnalysisResult(result);
        } else {
            addMessage('assistant', `Error: ${result.error}`);
        }
    } catch (error) {
        addMessage('assistant', `Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline-block';
        btn.querySelector('.btn-loading').style.display = 'none';
    }
}

function addMessage(role, content) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = role === 'user' ? '👤 You' : '🤖 Assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addAnalysisResult(result) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    let html = '<div class="message-header">🤖 Assistant</div><div class="message-content">';
    
    // Learned content badge
    if (result.has_learned_content) {
        html += `<div class="learning-badge">🧠 Using ${result.learned_doc_count} learned document(s)</div>`;
    }
    
    // Plan metrics
    html += '<h3>📋 Analysis Plan</h3>';
    html += '<div class="plan-metrics">';
    html += `<div class="metric-box"><div class="metric-label">Complexity</div><div class="metric-value">${result.plan.complexity}</div></div>`;
    html += `<div class="metric-box"><div class="metric-label">Retrieval K</div><div class="metric-value">${result.plan.retrieval_k}</div></div>`;
    html += `<div class="metric-box"><div class="metric-label">Depth</div><div class="metric-value">${result.plan.reasoning_depth}</div></div>`;
    html += '</div>';
    
    // Solution
    html += '<h3>💡 Solution & Analysis</h3>';
    html += `<div style="white-space: pre-wrap; line-height: 1.6;">${escapeHtml(result.reasoning)}</div>`;
    
    // Reflection
    html += '<h3>🔄 Quality Assessment</h3>';
    html += '<div class="reflection-box">';
    html += `<div><strong>Quality:</strong> <span class="quality-indicator quality-${result.reflection.quality}">${result.reflection.quality.toUpperCase()}</span></div>`;
    html += `<div><strong>Confidence:</strong> ${result.reflection.confidence.toUpperCase()}</div>`;
    html += `<div><strong>Completeness:</strong> ${result.reflection.completeness}%</div>`;
    if (result.reflection.gaps) {
        html += `<div><strong>Gaps:</strong> ${escapeHtml(result.reflection.gaps)}</div>`;
    }
    html += '</div>';
    
    // Reference links
    if (result.top_links && result.top_links.length > 0) {
        html += '<h3>🔗 Reference Links</h3>';
        html += '<div class="reference-links">';
        result.top_links.forEach((link, i) => {
            html += `<a href="${link}" target="_blank">${i + 1}. ${link}</a>`;
        });
        html += '</div>';
    }
    
    // Feedback section
    html += '<div class="feedback-section">';
    html += '<h3>📝 Was this solution helpful?</h3>';
    html += '<div class="feedback-buttons">';
    html += `<button class="feedback-btn success" onclick="submitFeedback('${result.session_id}', true)">✅ Yes, it worked!</button>`;
    html += `<button class="feedback-btn failed" onclick="showFeedbackForm('${result.session_id}')">❌ No, still having issues</button>`;
    html += '</div>';
    html += `<div id="feedback-form-${result.session_id}" class="feedback-form" style="display: none;">`;
    html += '<textarea id="feedback-text-${result.session_id}" placeholder="What happened? Please describe the issue..."></textarea>';
    html += `<button class="primary-btn" onclick="submitFeedback('${result.session_id}', false)" style="margin-top: 1rem;">Submit Feedback</button>`;
    html += '</div>';
    html += '</div>';
    
    html += '</div>';
    messageDiv.innerHTML = html;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showFeedbackForm(sessionId) {
    const form = document.getElementById(`feedback-form-${sessionId}`);
    if (form) {
        form.style.display = 'block';
    }
}

async function submitFeedback(sessionId, worked) {
    let newError = null;
    
    if (!worked) {
        const textarea = document.querySelector(`#feedback-form-${sessionId} textarea`);
        if (textarea) {
            newError = textarea.value.trim();
            if (!newError) {
                alert('Please provide some details to help the system learn');
                return;
            }
        }
    }
    
    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, worked, new_error: newError })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            addMessage('assistant', worked ? 
                '✅ Thank you! Your feedback helps improve the system! 🎉' :
                '📝 Feedback recorded! The system will learn from this. Go to the Learning Dashboard to apply the learning!'
            );
        } else {
            addMessage('assistant', `Error submitting feedback: ${result.error}`);
        }
    } catch (error) {
        addMessage('assistant', `Error: ${error.message}`);
    }
}

// Learning dashboard
async function loadLearningStats() {
    try {
        const response = await fetch('/api/learning/stats');
        const stats = await response.json();
        
        document.getElementById('total-feedback').textContent = stats.total_feedback;
        document.getElementById('success-rate').textContent = `${stats.success_rate.toFixed(1)}%`;
        document.getElementById('pending-updates').textContent = stats.pending_updates;
        document.getElementById('learned-docs').textContent = stats.learned_documents;
    } catch (error) {
        console.error('Error loading learning stats:', error);
    }
}

async function loadSystemStats() {
    try {
        const response = await fetch('/api/system/stats');
        const stats = await response.json();
        
        document.getElementById('total-docs').textContent = stats.total_docs;
        document.getElementById('graph-nodes').textContent = stats.graph_nodes;
        document.getElementById('graph-edges').textContent = stats.graph_edges;
    } catch (error) {
        console.error('Error loading system stats:', error);
    }
}

document.getElementById('apply-learning-btn').addEventListener('click', async () => {
    const btn = document.getElementById('apply-learning-btn');
    const resultDiv = document.getElementById('learning-result');
    
    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loading').style.display = 'inline-block';
    
    try {
        const response = await fetch('/api/learning/apply', { method: 'POST' });
        const result = await response.json();
        
        if (response.ok) {
            if (result.updated === false) {
                resultDiv.innerHTML = '<div class="success-message">✨ System is fully up-to-date! No pending learning items.</div>';
            } else {
                resultDiv.innerHTML = `
                    <div class="success-message">
                        <h3>✅ ReAct Learning Applied Successfully!</h3>
                        <p>Documents Added: ${result.docs_added}</p>
                        <p>Graph Nodes Added: ${result.graph_nodes_added}</p>
                        <p>Graph Edges Added: ${result.graph_edges_added}</p>
                    </div>
                `;
                loadLearningStats();
                loadSystemStats();
            }
        } else {
            resultDiv.innerHTML = `<div class="error-message">Error: ${result.error}</div>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline-block';
        btn.querySelector('.btn-loading').style.display = 'none';
    }
});

// Initial load
loadLearningStats();
loadSystemStats();