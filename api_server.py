"""
BuzzStreet – api_server.py
Standalone Production REST API Server.
Exposes HTTP endpoints for external trading bots, mobile apps, and administrative clients:
- GET  /api/health            : System Health & Component Telemetry
- GET  /api/narrative-events  : Recent Narrative Shift Events Log
- POST /api/auth/send-otp     : Trigger Twilio Carrier SMS OTP
- POST /api/auth/verify-otp   : Verify SMS OTP Code
- POST /api/analyze-headline  : Analyze Headline Sentiment & Phase
"""

import json
import http.server
import socketserver
import urllib.parse
import system_health
import db
import auth
import narrative_detector
from nlp_pipeline import preprocess_headline_detailed
from ml_model import model_instance

PORT = 8000

class BuzzStreetAPIHandler(http.server.BaseHTTPRequestHandler):
    def _send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        if path == "/api/health" or path == "/health":
            health = system_health.get_system_health_status()
            self._send_json_response(health)
        elif path == "/api/narrative-events":
            events = db.get_narrative_events(limit=20)
            self._send_json_response({"status": "success", "count": len(events), "events": events})
        else:
            self._send_json_response({"error": "Endpoint not found", "available_endpoints": ["/api/health", "/api/narrative-events", "/api/analyze-headline"]}, status_code=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        
        try:
            body = json.loads(post_body)
        except Exception:
            body = {}

        if path == "/api/analyze-headline":
            headline = body.get("headline", "")
            if not headline:
                self._send_json_response({"error": "Missing 'headline' field in JSON request body."}, status_code=400)
                return
                
            v_res = narrative_detector.analyze_vader_sentiment(headline)
            lr_res = narrative_detector.predict_lr_sentiment(headline, model_instance.vectorizer, model_instance.model)
            composite = narrative_detector.calculate_composite_index([v_res["compound"]], [lr_res], vader_weight=0.50)
            phase = narrative_detector.detect_narrative_phase(composite)
            
            self._send_json_response({
                "status": "success",
                "headline": headline,
                "vader_sentiment": v_res,
                "lr_sentiment": lr_res,
                "composite_score": round(composite, 4),
                "narrative_phase": phase
            })
        elif path == "/api/auth/send-otp":
            phone = body.get("phone", "")
            success, msg = auth.send_otp_backend(phone)
            self._send_json_response({"status": "success" if success else "error", "message": msg})
        elif path == "/api/auth/verify-otp":
            otp = body.get("otp", "")
            success, msg = auth.verify_otp_backend(otp)
            self._send_json_response({"status": "success" if success else "error", "message": msg})
        else:
            self._send_json_response({"error": "Endpoint not found"}, status_code=404)

def run_api_server(port=PORT):
    with socketserver.TCPServer(("", port), BuzzStreetAPIHandler) as httpd:
        print(f"🚀 BuzzStreet Production REST API Server running on http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_api_server()
