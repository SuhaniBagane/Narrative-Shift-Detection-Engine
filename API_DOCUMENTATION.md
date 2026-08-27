# BuzzStreet API Documentation

## Authentication Endpoints

### `POST /api/auth/send-otp`
Triggers real SMS OTP dispatch to an international phone number via Twilio Verify API.
- **Request Body:**
  ```json
  {
    "phone": "+917676526744"
  }
  ```
- **Response (Success):**
  ```json
  {
    "status": "success",
    "message": "📲 Real SMS OTP dispatched to +91 ******6744."
  }
  ```

### `POST /api/auth/verify-otp`
Verifies entered 6-digit OTP code against the backend SMS provider.
- **Request Body:**
  ```json
  {
    "phone": "+917676526744",
    "otp": "483921"
  }
  ```
- **Response (Success):**
  ```json
  {
    "status": "approved",
    "message": "✅ OTP Verification Successful!"
  }
  ```

---

## System Health Diagnostic Endpoint

### `GET /api/health`
Returns real-time component health checks and telemetry metrics.
- **Response:**
  ```json
  {
    "status_code": 200,
    "timestamp": "2026-08-28 00:25:00 UTC",
    "latency_ms": 1.25,
    "services": {
      "Database Service": "🟢 Online",
      "NLP Pipeline Engine": "🟢 Online",
      "ML Sentiment Classifier": "🟢 Online (v2.1)",
      "Market Data API": "🟢 Online (Yahoo Finance)",
      "News Ingestion API": "🟢 Online (Ingesting)",
      "Forecast Engine": "🟢 Online",
      "Authentication Gate": "🟢 Online"
    },
    "telemetry": {
      "articles_processed_today": 142,
      "total_database_articles": 300000,
      "api_failure_rate": "0.0%",
      "stale_data_warnings": 0
    }
  }
  ```
