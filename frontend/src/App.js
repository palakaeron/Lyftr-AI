import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';
function generateSessionId() {
  return 'sess-' + Math.random().toString(36).slice(2, 11) + '-' + Date.now();
}

// ── Metadata badge ─────────────────────────────────────────────────────────────
function MetaBadge({ meta }) {
  if (!meta) return null;
  const isCache = meta.resolution_source === 'LLM_CACHE';
  return (
    <div className="meta-badge">
      <span
        data-testid="meta-source"
        className={`meta-source ${isCache ? 'cache' : 'live'}`}
      >
        {meta.resolution_source}
      </span>
      <span className="meta-sep">·</span>
      <span data-testid="meta-model" className="meta-item">
        {meta.model}
      </span>
      <span className="meta-sep">·</span>
      <span data-testid="meta-latency" className="meta-item">
        {meta.latency_ms} ms
      </span>
      <span className="meta-sep">·</span>
      <span data-testid="meta-input-tokens" className="meta-item">
        ↑ {meta.input_tokens} tok
      </span>
      <span className="meta-sep">·</span>
      <span data-testid="meta-output-tokens" className="meta-item">
        ↓ {meta.output_tokens} tok
      </span>
      {meta.key_id && (
        <>
          <span className="meta-sep">·</span>
          <span className="meta-item key-id">{meta.key_id}</span>
        </>
      )}
      {meta.fallback_attempts > 0 && (
        <>
          <span className="meta-sep">·</span>
          <span className="meta-item fallback">
            {meta.fallback_attempts} fallback{meta.fallback_attempts > 1 ? 's' : ''}
          </span>
        </>
      )}
    </div>
  );
}

// ── Message bubble ──────────────────────────────────────────────────────────────
function MessageBubble({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="message-row user">
        <div className="bubble user-bubble">{msg.content}</div>
      </div>
    );
  }

  if (msg.role === 'error') {
    return (
      <div className="message-row assistant">
        <div className="bubble error-bubble">{msg.content}</div>
      </div>
    );
  }

  return (
    <div className="message-row assistant">
      <div data-testid="assistant-message" className="assistant-message-wrapper">
        <div className="bubble assistant-bubble">{msg.content}</div>
        <MetaBadge meta={msg.metadata} />
      </div>
    </div>
  );
}

// ── Loading indicator ──────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="message-row assistant">
      <div className="bubble assistant-bubble typing">
        <span /><span /><span />
      </div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [sessionId, setSessionId] = useState(() => generateSessionId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [pastSessions, setPastSessions] = useState([]);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  // Fetch the full session list from the backend
  const loadSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      if (!res.ok) return;
      const data = await res.json();
      setPastSessions(data.sessions || []);
    } catch {
      // Sidebar list is non-critical; silently ignore network errors
    }
  }, []);

  // Load session list once on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Load existing session history on mount (and whenever sessionId changes)
  useEffect(() => {
    let cancelled = false;
    setHistoryLoading(true);
    setMessages([]);

    async function loadHistory() {
      try {
        const res = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/messages`);
        if (cancelled) return;
        if (res.status === 404) {
          // New session — no history to load
          return;
        }
        if (!res.ok) {
          console.warn('Failed to load session history:', res.status);
          return;
        }
        const data = await res.json();
        if (cancelled) return;
        const loaded = (data.messages || []).map((m) => ({
          role: m.role,
          content: m.content,
          metadata: m.metadata || null,
        }));
        setMessages(loaded);
      } catch (err) {
        if (!cancelled) console.warn('Could not load session history:', err);
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    }

    loadHistory();
    return () => { cancelled = true; };
  }, [sessionId]);

  // Switch to a past session
  const switchSession = useCallback((id) => {
    if (id === sessionId) return;
    setSessionId(id);
    setInput('');
    setSystemPrompt('');
  }, [sessionId]);

  // Start a new session: generate fresh ID, clear state
  const startNewSession = useCallback(() => {
    const fresh = generateSessionId();
    setSessionId(fresh);
    setInput('');
    setSystemPrompt('');
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const body = { session_id: sessionId, message: text };
      if (systemPrompt.trim()) body.system_prompt = systemPrompt.trim();

      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (res.status === 422) {
        const data = await res.json();
        const detail =
          typeof data.detail === 'string'
            ? data.detail
            : Array.isArray(data.detail)
            ? data.detail.map((e) => e.msg || JSON.stringify(e)).join(', ')
            : 'Invalid request.';
        setMessages((prev) => [
          ...prev,
          { role: 'error', content: `Validation error: ${detail}` },
        ]);
        return;
      }

      if (res.status === 503) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'error',
            content: 'All AI providers are currently unavailable. Please try again later.',
          },
        ]);
        return;
      }

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setMessages((prev) => [
          ...prev,
          {
            role: 'error',
            content: data.detail || `Server error (${res.status}). Please try again.`,
          },
        ]);
        return;
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.reply,
          metadata: data.metadata,
        },
      ]);
      // Refresh sidebar so new sessions appear immediately
      loadSessions();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'error',
          content: 'Could not reach the server. Check your connection.',
        },
      ]);
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  }, [input, loading, sessionId, systemPrompt, loadSessions]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="layout">

      {/* ── Sidebar ──────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <button
            data-testid="new-session-button"
            className="new-chat-btn"
            onClick={startNewSession}
            title="Start a new session"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New Chat
          </button>
        </div>

        <div className="sidebar-section-label">Recent chats</div>
        <div className="sidebar-sessions">
          {pastSessions.length === 0 && (
            <p className="sidebar-empty">No previous chats yet.</p>
          )}
          {pastSessions.map((s) => (
            <button
              key={s.session_id}
              className={`session-item${s.session_id === sessionId ? ' active' : ''}`}
              onClick={() => switchSession(s.session_id)}
              title={s.session_id}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              <span className="session-item-label">
                {s.preview || s.session_id.slice(0, 20) + '…'}
              </span>
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <span className="sidebar-logo-icon">⚡</span>
          <span className="sidebar-logo-text">Lyftr AI</span>
        </div>
      </aside>

      {/* ── Main column ──────────────────────────────────────────── */}
      <div className="main-col">

        {/* Top bar */}
        <div className="topbar">
          <span className="topbar-title" title={sessionId}>
            {sessionId}
          </span>
        </div>

        {/* System prompt */}
        <div className="system-prompt-bar">
          <textarea
            data-testid="system-prompt-input"
            className="system-prompt-input"
            placeholder="System prompt (optional) — e.g. You are a helpful assistant."
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={2}
          />
        </div>

        {/* Chat area */}
        <main className="chat-area">
          {historyLoading && (
            <div className="history-loading">Loading conversation history…</div>
          )}
          {!historyLoading && messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">⚡</div>
              <p className="empty-title">How can I help you today?</p>
              <p className="empty-sub">Type a message below to get started.</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </main>

        {/* Composer */}
        <div className="input-bar">
          <div className="input-bar-inner">
            <div className="input-wrapper">
              <textarea
                ref={textareaRef}
                data-testid="chat-input"
                className="chat-input"
                placeholder="Message Lyftr AI…"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={loading}
              />
              <button
                data-testid="send-button"
                className="send-button"
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                aria-label="Send message"
              >
                {loading ? (
                  <span className="spinner" />
                ) : (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                )}
              </button>
            </div>
            <p className="composer-hint">Enter to send · Shift+Enter for new line</p>
          </div>
        </div>

      </div>{/* /main-col */}
    </div>
  );
}
