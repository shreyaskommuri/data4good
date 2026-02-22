import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Bot, User, Loader, MapPin } from 'lucide-react';
import { sendChat } from '../api';

function Message({ msg }) {
  const isBot = msg.role === 'assistant';
  return (
    <div style={{
      display: 'flex',
      gap: 8,
      alignItems: 'flex-start',
      justifyContent: isBot ? 'flex-start' : 'flex-end',
      marginBottom: 12,
    }}>
      {isBot && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
          background: 'linear-gradient(135deg, #3b82f6, #22d3ee)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Bot size={14} color="#fff" />
        </div>
      )}
      <div style={{
        maxWidth: '78%',
        padding: '8px 12px',
        borderRadius: isBot ? '4px 12px 12px 12px' : '12px 4px 12px 12px',
        background: isBot
          ? 'rgba(59, 130, 246, 0.12)'
          : 'rgba(34, 211, 238, 0.12)',
        border: isBot
          ? '1px solid rgba(59, 130, 246, 0.25)'
          : '1px solid rgba(34, 211, 238, 0.25)',
        color: '#e8e8f0',
        fontSize: '13px',
        lineHeight: 1.55,
        whiteSpace: 'pre-wrap',
      }}>
        {msg.content}
      </div>
      {!isBot && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
          background: 'rgba(34, 211, 238, 0.15)',
          border: '1px solid rgba(34, 211, 238, 0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <User size={14} color="#22d3ee" />
        </div>
      )}
    </div>
  );
}

export default function ChatPanel({ params, simData, selectedTract, allTracts }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hi! I'm your Santa Barbara County resilience analyst. I have data on all 109 census tracts loaded — ask me about any tract by name, compare risk levels, or explore policy options.\n\nTip: click a tract on the map to set it as the focus for deeper context.",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, open]);

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setLoading(true);
    try {
      const data = await sendChat(text, selectedTract || {}, params || {}, simData || {}, allTracts || []);
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I could not reach the backend. Make sure the API server is running.',
      }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <>
      {/* Floating toggle button */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          position: 'fixed',
          bottom: 28,
          right: 28,
          width: 52,
          height: 52,
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #3b82f6, #22d3ee)',
          border: 'none',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 20px rgba(59,130,246,0.4)',
          zIndex: 1000,
          transition: 'transform 0.2s',
        }}
        onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.1)'}
        onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
      >
        {open ? <X size={22} color="#fff" /> : <MessageCircle size={22} color="#fff" />}
      </button>

      {/* Chat panel */}
      {open && (
        <div style={{
          position: 'fixed',
          bottom: 90,
          right: 28,
          width: 380,
          height: 520,
          borderRadius: 16,
          background: 'rgba(14,16,22,0.97)',
          border: '1px solid rgba(59,130,246,0.25)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 999,
          overflow: 'hidden',
        }}>
          {/* Header */}
          <div style={{
            padding: '14px 16px',
            background: 'linear-gradient(135deg, rgba(59,130,246,0.15), rgba(34,211,238,0.08))',
            borderBottom: '1px solid rgba(59,130,246,0.2)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
          }}>
            <div style={{
              width: 32, height: 32, borderRadius: '50%',
              background: 'linear-gradient(135deg, #3b82f6, #22d3ee)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Bot size={16} color="#fff" />
            </div>
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#e8e8f0' }}>
                Resilience Analyst
              </div>
              <div style={{ fontSize: '11px', color: '#22d3ee' }}>
                {selectedTract ? selectedTract.name : 'No tract selected'} · GPT-4o mini
              </div>
            </div>
          </div>

          {/* Messages */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '14px 14px 0',
          }}>
            {/* No tract selected — soft hint only */}
          {!selectedTract && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 12px',
              marginBottom: 10,
              borderRadius: 8,
              background: 'rgba(59,130,246,0.06)',
              border: '1px solid rgba(59,130,246,0.15)',
              fontSize: '12px',
              color: '#93c5fd',
            }}>
              <MapPin size={12} style={{ flexShrink: 0 }} />
              Click a tract on the map to set it as focus
            </div>
          )}

          {/* Messages */}
          {messages.map((msg, i) => <Message key={i} msg={msg} />)}
            {loading && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: '50%',
                  background: 'linear-gradient(135deg, #3b82f6, #22d3ee)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <Bot size={14} color="#fff" />
                </div>
                <div style={{
                  padding: '8px 14px',
                  borderRadius: '4px 12px 12px 12px',
                  background: 'rgba(59,130,246,0.12)',
                  border: '1px solid rgba(59,130,246,0.25)',
                }}>
                  <Loader size={14} color="#3b82f6" style={{ animation: 'spin 1s linear infinite' }} />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div style={{
            padding: '10px 12px',
            borderTop: '1px solid rgba(59,130,246,0.15)',
            display: 'flex',
            gap: 8,
            alignItems: 'flex-end',
          }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask about policy, risks, recovery..."
              rows={1}
              style={{
                flex: 1,
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(59,130,246,0.2)',
                borderRadius: 10,
                padding: '8px 12px',
                color: '#e8e8f0',
                fontSize: '13px',
                resize: 'none',
                outline: 'none',
                fontFamily: 'inherit',
                lineHeight: 1.4,
              }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              style={{
                width: 36, height: 36,
                borderRadius: 10,
                background: input.trim() && !loading
                  ? 'linear-gradient(135deg, #3b82f6, #22d3ee)'
                  : 'rgba(255,255,255,0.05)',
                border: 'none',
                cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                transition: 'background 0.2s',
                flexShrink: 0,
              }}
            >
              <Send size={15} color={input.trim() && !loading ? '#fff' : '#555'} />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
