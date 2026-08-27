# BuzzStreet Security & Authorization Documentation

## Security Principles Enforced
1. **Zero Frontend Secret Exposure:** No API keys, database credentials, or Twilio tokens are exposed in client JavaScript or state.
2. **Environment Variable Secret Management:** All credentials are loaded via `os.getenv` from `.env`.
3. **Session Authorization Guards:** Every protected dashboard route validates `st.session_state.authenticated == True` before rendering.
4. **Input Validation & Sanitize:** E.164 phone number formatting and SHA-256 deduplication hashing.
5. **Rate Limiting:** 60-second cooldown period between SMS OTP resend requests.
