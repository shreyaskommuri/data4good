import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: 40,
          background: '#0a0b0f',
          color: '#e8e8f0',
          minHeight: '100vh',
          fontFamily: 'monospace',
        }}>
          <h2 style={{ color: '#f43f5e', marginBottom: 16 }}>
            Something went wrong
          </h2>
          <pre style={{
            background: '#181924',
            padding: 20,
            borderRadius: 12,
            overflow: 'auto',
            fontSize: 14,
            lineHeight: 1.6,
            border: '1px solid rgba(244,63,94,0.3)',
          }}>
            {this.state.error?.toString()}
            {'\n\n'}
            {this.state.errorInfo?.componentStack}
          </pre>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: 16, padding: '10px 24px', borderRadius: 8,
              background: '#3b82f6', color: '#fff', border: 'none',
              cursor: 'pointer', fontSize: 14, fontWeight: 600,
            }}
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
