import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { HelpCircle } from 'lucide-react';

export default function Tooltip({ content }) {
  const [show, setShow] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0, pos: 'top' });
  const iconRef = useRef(null);
  const tipRef = useRef(null);

  const updatePosition = useCallback(() => {
    if (!iconRef.current) return;
    const rect = iconRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const iconTop = rect.top;
    const iconBottom = rect.bottom;

    // Default: show above. If not enough room, show below.
    let pos = 'top';
    let y = iconTop - 10; // above the icon
    if (iconTop < 160) {
      pos = 'bottom';
      y = iconBottom + 10;
    }

    setCoords({ x: centerX, y, pos });
  }, []);

  const handleMouseEnter = useCallback(() => {
    updatePosition();
    setShow(true);
  }, [updatePosition]);

  const handleMouseLeave = useCallback(() => {
    setShow(false);
  }, []);

  // Recalculate position on scroll while visible
  useEffect(() => {
    if (!show) return;
    const onScroll = () => updatePosition();
    window.addEventListener('scroll', onScroll, true);
    return () => window.removeEventListener('scroll', onScroll, true);
  }, [show, updatePosition]);

  return (
    <span
      ref={iconRef}
      style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', marginLeft: 8, zIndex: 10 }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
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
          border: `1.5px solid ${show ? 'rgba(167,139,250,0.6)' : 'rgba(255,255,255,0.3)'}`,
          background: show ? 'rgba(167,139,250,0.2)' : 'rgba(255,255,255,0.1)',
          color: show ? '#a78bfa' : '#9ca3af',
          cursor: 'help',
          transition: 'all 0.2s ease',
          flexShrink: 0,
        }}
      >
        <HelpCircle size={13} strokeWidth={2.5} />
      </span>

      {/* Tooltip rendered via portal to escape overflow:hidden containers */}
      {show && createPortal(
        <div
          ref={tipRef}
          style={{
            position: 'fixed',
            left: coords.x,
            ...(coords.pos === 'top'
              ? { bottom: `calc(100vh - ${coords.y}px)` }
              : { top: coords.y }),
            transform: 'translateX(-50%)',
            width: 300,
            padding: '14px 16px',
            background: '#1a1b2e',
            border: '1px solid rgba(167,139,250,0.2)',
            borderRadius: 12,
            fontSize: '0.82rem',
            color: '#c8c9d8',
            lineHeight: 1.7,
            zIndex: 99999,
            boxShadow: '0 12px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(167,139,250,0.08)',
            pointerEvents: 'none',
            animation: 'tooltipFadeIn 0.15s ease-out',
          }}
        >
          {/* Arrow */}
          <div style={{
            position: 'absolute',
            ...(coords.pos === 'top'
              ? { bottom: -6, top: 'auto' }
              : { top: -6, bottom: 'auto' }),
            left: '50%',
            transform: `translateX(-50%) rotate(${coords.pos === 'top' ? '45deg' : '225deg'})`,
            width: 10,
            height: 10,
            background: '#1a1b2e',
            borderRight: '1px solid rgba(167,139,250,0.2)',
            borderBottom: '1px solid rgba(167,139,250,0.2)',
          }} />
          <div dangerouslySetInnerHTML={{ __html: content }} />
        </div>,
        document.body
      )}

      <style>{`
        @keyframes tooltipFadeIn {
          from { opacity: 0; transform: translateX(-50%) translateY(${coords.pos === 'top' ? '4px' : '-4px'}); }
          to   { opacity: 1; transform: translateX(-50%) translateY(0); }
        }
      `}</style>
    </span>
  );
}
