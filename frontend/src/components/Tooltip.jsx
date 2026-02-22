import React, { useState, useRef, useEffect } from 'react';
import { HelpCircle } from 'lucide-react';

export default function Tooltip({ content }) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState('bottom'); // 'bottom' or 'top'
  const tipRef = useRef(null);
  const iconRef = useRef(null);

  useEffect(() => {
    if (show && tipRef.current) {
      const rect = tipRef.current.getBoundingClientRect();
      // If tooltip goes above viewport, flip to below
      if (rect.top < 8) {
        setPos('bottom');
      }
    }
  }, [show]);

  return (
    <span
      ref={iconRef}
      style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', marginLeft: 8 }}
      onMouseEnter={() => { setPos('top'); setShow(true); }}
      onMouseLeave={() => setShow(false)}
    >
      {/* Icon button */}
      <span
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 20,
          height: 20,
          borderRadius: '50%',
          border: `1.5px solid ${show ? 'rgba(167,139,250,0.6)' : 'rgba(255,255,255,0.25)'}`,
          background: show ? 'rgba(167,139,250,0.15)' : 'rgba(255,255,255,0.06)',
          color: show ? '#a78bfa' : '#8b8ca0',
          cursor: 'help',
          transition: 'all 0.2s ease',
          flexShrink: 0,
        }}
      >
        <HelpCircle size={13} strokeWidth={2.5} />
      </span>

      {/* Tooltip popup */}
      {show && (
        <div
          ref={tipRef}
          style={{
            position: 'absolute',
            ...(pos === 'top'
              ? { bottom: 'calc(100% + 10px)', top: 'auto' }
              : { top: 'calc(100% + 10px)', bottom: 'auto' }),
            left: '50%',
            transform: 'translateX(-50%)',
            width: 300,
            padding: '14px 16px',
            background: '#1a1b2e',
            border: '1px solid rgba(167,139,250,0.2)',
            borderRadius: 12,
            fontSize: '0.82rem',
            color: '#c8c9d8',
            lineHeight: 1.7,
            zIndex: 1000,
            boxShadow: '0 12px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(167,139,250,0.08)',
            pointerEvents: 'none',
            animation: 'tooltipFadeIn 0.15s ease-out',
          }}
        >
          {/* Arrow */}
          <div style={{
            position: 'absolute',
            ...(pos === 'top'
              ? { bottom: -6, top: 'auto' }
              : { top: -6, bottom: 'auto' }),
            left: '50%',
            transform: `translateX(-50%) rotate(${pos === 'top' ? '45deg' : '225deg'})`,
            width: 10,
            height: 10,
            background: '#1a1b2e',
            borderRight: '1px solid rgba(167,139,250,0.2)',
            borderBottom: '1px solid rgba(167,139,250,0.2)',
          }} />
          <div dangerouslySetInnerHTML={{ __html: content }} />
        </div>
      )}

      <style>{`
        @keyframes tooltipFadeIn {
          from { opacity: 0; transform: translateX(-50%) translateY(${pos === 'top' ? '4px' : '-4px'}); }
          to   { opacity: 1; transform: translateX(-50%) translateY(0); }
        }
      `}</style>
    </span>
  );
}
