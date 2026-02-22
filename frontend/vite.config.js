import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Raise the chunk size warning limit (we're aware of large vendors)
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        // Split vendors into separate cached chunks so browsers only
        // re-download what changed and can load chunks in parallel.
        manualChunks: {
          'vendor-react':    ['react', 'react-dom'],
          'vendor-map':      ['maplibre-gl'],
          'vendor-deck':     ['deck.gl', '@deck.gl/layers', '@deck.gl/react'],
          'vendor-nivo':     ['@nivo/core', '@nivo/sankey', '@nivo/chord',
                              '@nivo/circle-packing', '@nivo/tooltip'],
          'vendor-recharts': ['recharts'],
          'vendor-pdf':      ['jspdf', 'html2canvas'],
          'vendor-lucide':   ['lucide-react'],
        },
      },
    },
  },
})
