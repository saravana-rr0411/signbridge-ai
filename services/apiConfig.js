// services/apiConfig.js - Centralized API Endpoint Configuration
// Reads VITE_API_BASE_URL from environment with fallback to http://127.0.0.1:8000

export const API_BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL)
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://127.0.0.1:8000';

export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/health`,
  LABELS: `${API_BASE_URL}/labels`,
  LABELS_V3_SIX_SIGN: `${API_BASE_URL}/labels/v3-six-sign`,
  LABELS_V6_10_SIGN: `${API_BASE_URL}/labels/v6-10-sign`,
  PREDICT_SEQUENCE: `${API_BASE_URL}/predict/sequence`,
  PREDICT_SEQUENCE_V3_SIX_SIGN: `${API_BASE_URL}/predict/sequence/v3-six-sign`,
  PREDICT_SEQUENCE_V6_10_SIGN: `${API_BASE_URL}/predict/sequence/v6-10-sign`,
  PREDICT_STATIC: `${API_BASE_URL}/predict/static`
};
