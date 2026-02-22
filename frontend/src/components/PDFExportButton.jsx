import React, { useState } from 'react';
import { FileDown, Loader, ChevronDown } from 'lucide-react';
import { generatePDFReport } from '../utils/pdfExport';

export default function PDFExportButton({ params, simData, tracts, selectedTract, onSelectTract }) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);

  const handleGeneratePDF = async () => {
    if (!selectedTract) {
      alert('Please select a tract first');
      return;
    }
    
    setIsGenerating(true);
    try {
      await generatePDFReport(params, simData, selectedTract);
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000);
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Failed to generate PDF report');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      gap: '12px',
      alignItems: 'center',
      flexWrap: 'wrap',
      justifyContent: 'center'
    }}>
      {/* Tract Selector */}
      <div style={{ position: 'relative' }}>
        <button
          onClick={() => setShowDropdown(!showDropdown)}
          style={{
            padding: '10px 16px',
            backgroundColor: 'rgba(34, 211, 238, 0.1)',
            border: '1px solid rgba(34, 211, 238, 0.3)',
            borderRadius: '8px',
            color: '#22d3ee',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(34, 211, 238, 0.2)'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgba(34, 211, 238, 0.1)'}
        >
          {selectedTract ? selectedTract.name : 'Select Tract'}
          <ChevronDown size={16} />
        </button>

        {showDropdown && tracts && (
          <div style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            marginTop: '8px',
            backgroundColor: 'rgba(18, 19, 26, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '8px',
            minWidth: '200px',
            maxHeight: '300px',
            overflowY: 'auto',
            zIndex: 1000,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
          }}>
            {tracts.map((tract) => (
              <button
                key={tract.tract_id || tract.name}
                onClick={() => {
                  onSelectTract(tract);
                  setShowDropdown(false);
                }}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  backgroundColor: selectedTract && selectedTract.name === tract.name ? 'rgba(34, 211, 238, 0.2)' : 'transparent',
                  border: 'none',
                  color: '#e8e8f0',
                  cursor: 'pointer',
                  fontSize: '13px',
                  textAlign: 'left',
                  borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                  transition: 'background-color 0.2s ease'
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(34, 211, 238, 0.15)'}
                onMouseLeave={(e) => selectedTract && selectedTract.name === tract.name 
                  ? e.currentTarget.style.backgroundColor = 'rgba(34, 211, 238, 0.2)'
                  : e.currentTarget.style.backgroundColor = 'transparent'
                }
              >
                <div style={{ fontWeight: '500' }}>{tract.name}</div>
                <div style={{ fontSize: '11px', color: 'rgba(232, 232, 240, 0.6)' }}>
                  Risk: {((tract.exodus_prob || 0) * 100).toFixed(1)}% | EJ: {tract.ej_percentile || 0}%
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Generate PDF Button */}
      <button
        onClick={handleGeneratePDF}
        disabled={isGenerating || !selectedTract}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '10px 16px',
          backgroundColor: selectedTract ? 'rgba(59, 130, 246, 0.1)' : 'rgba(90, 91, 112, 0.1)',
          border: selectedTract ? '1px solid rgba(59, 130, 246, 0.3)' : '1px solid rgba(90, 91, 112, 0.2)',
          borderRadius: '8px',
          cursor: isGenerating || !selectedTract ? 'not-allowed' : 'pointer',
          opacity: isGenerating || !selectedTract ? 0.5 : 1,
          transition: 'all 0.2s ease',
          fontSize: '14px',
          fontWeight: '600',
          color: selectedTract ? '#3b82f6' : '#5a5b70'
        }}
        onMouseEnter={(e) => {
          if (!isGenerating && selectedTract) {
            e.currentTarget.style.backgroundColor = 'rgba(59, 130, 246, 0.2)';
          }
        }}
        onMouseLeave={(e) => {
          if (selectedTract) {
            e.currentTarget.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
          }
        }}
      >
        {isGenerating ? (
          <>
            <Loader size={16} style={{ animation: 'spin 1s linear infinite' }} />
            <span>Generating...</span>
          </>
        ) : showSuccess ? (
          <>
            <span style={{ color: '#34d399' }}>✓ PDF Downloaded</span>
          </>
        ) : (
          <>
            <FileDown size={16} />
            <span>Generate PDF Report</span>
          </>
        )}
      </button>
    </div>
  );
}

// CSS for spin animation
const style = document.createElement('style');
style.innerHTML = `
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;
if (!document.querySelector('style[data-pdf-animation]')) {
  style.setAttribute('data-pdf-animation', 'true');
  document.head.appendChild(style);
}
