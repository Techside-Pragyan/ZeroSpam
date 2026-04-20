import React, { useState, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, ShieldAlert, Upload, Send, Trash2, Info, CheckCircle2, AlertCircle } from 'lucide-react';

const API_URL = 'http://localhost:8000';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/predict`, { text });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData);
      setResult(response.data);
      // If result is successful, update the text area with a snippet or filename
      setText(`[Analysis of ${file.name}]`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Unsupported file format');
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header>
        <motion.h1 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          ZeroSpam AI
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Advanced Neural Email Classification System
        </motion.p>
      </header>

      <main>
        <motion.section 
          className="glass-panel"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          <div className="input-section">
            <div className="text-area-wrapper">
              <textarea
                placeholder="Paste your email content or message here for instant analysis..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>

            <div className="action-buttons">
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload} 
                style={{ display: 'none' }} 
                accept=".txt,.eml"
              />
              <button className="btn-secondary" onClick={() => fileInputRef.current.click()}>
                <Upload size={18} /> Upload file
              </button>
              <button className="btn-secondary" onClick={clearAll}>
                <Trash2 size={18} /> Clear
              </button>
              <button className="btn-primary" onClick={handlePredict} disabled={loading || !text.trim()}>
                {loading ? <div className="loader" /> : <><Send size={18} /> Analyze Message</>}
              </button>
            </div>
          </div>
        </motion.section>

        <AnimatePresence>
          {error && (
            <motion.div 
              className="glass-panel"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              style={{ borderColor: 'var(--accent-spam)', color: '#fda4af', display: 'flex', alignItems: 'center', gap: '10px' }}
            >
              <AlertCircle size={20} />
              {error}
            </motion.div>
          )}

          {result && (
            <motion.section 
              className="glass-panel"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className="results-container">
                <div className="result-card">
                  <div className={`status-badge ${result.is_spam ? 'spam-text' : 'ham-text'}`}>
                    {result.is_spam ? <ShieldAlert size={48} /> : <CheckCircle2 size={48} />}
                    <div style={{ marginTop: '0.5rem' }}>{result.label}</div>
                  </div>
                  
                  <div style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Confidence</span>
                      <span style={{ fontWeight: 600 }}>{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="confidence-meter">
                      <motion.div 
                        className="confidence-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence * 100}%` }}
                        style={{ background: result.is_spam ? 'var(--accent-spam)' : 'var(--accent-ham)' }}
                      />
                    </div>
                  </div>
                </div>

                <div className="explanation-section">
                  <h3>
                    <Info size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
                    Explainability Report
                  </h3>
                  <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                    {result.is_spam 
                      ? "The following patterns and keywords strongly suggest this message is unsolicited or potentially harmful."
                      : "The message patterns align with legitimate communication styles."}
                  </p>
                  
                  <div className="word-chips">
                    {result.explanation.length > 0 ? (
                      result.explanation.map((item, i) => (
                        <motion.div 
                          key={i}
                          className={`word-chip ${item.impact.toLowerCase()}`}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.05 }}
                        >
                          {item.word} 
                          <span style={{ opacity: 0.6, fontSize: '0.7rem' }}>
                            {item.weight > 0 ? '+' : ''}{item.weight.toFixed(2)}
                          </span>
                        </motion.div>
                      ))
                    ) : (
                      <p style={{ fontStyle: 'italic', color: 'var(--text-secondary)' }}>No significant spam triggers detected.</p>
                    )}
                  </div>
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </main>

      <footer style={{ textAlign: 'center', marginTop: '4rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
        &copy; 2026 ZeroSpam AI. Powered by Scikit-Learn NLP.
      </footer>
    </div>
  );
}

export default App;
