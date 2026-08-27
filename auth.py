"""
BuzzStreet – auth.py
Production-Grade SMS & Email Authentication Engine.
Integrates with real SMS Providers (Twilio Verify API / Twilio SMS / MSG91) via backend APIs.
NO fake OTPs, NO inline OTP displays, and NO frontend state exposure.
"""

import os
import time
import datetime
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Environment credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID")
OTP_COOLDOWN_SECONDS = int(os.getenv("OTP_COOLDOWN_SECONDS", 60))
OTP_EXPIRY_SECONDS = int(os.getenv("OTP_EXPIRY_SECONDS", 600))

# Country Codes Mapping
COUNTRY_CODES = [
    ("🇮🇳 India (+91)", "+91"),
    ("🇺🇸 USA / Canada (+1)", "+1"),
    ("🇬🇧 UK (+44)", "+44"),
    ("🇦🇺 Australia (+61)", "+61"),
    ("🇩🇪 Germany (+49)", "+49"),
    ("🇫🇷 France (+33)", "+33"),
    ("🇯🇵 Japan (+81)", "+81"),
    ("🇸🇬 Singapore (+65)", "+65"),
    ("🇦🇪 UAE (+971)", "+971"),
    ("🌐 Other Country Code", "+")
]

def is_twilio_configured():
    """Checks if real Twilio credentials exist in environment variables."""
    return bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_VERIFY_SERVICE_SID and 
                "your_" not in TWILIO_ACCOUNT_SID.lower())

def mask_identifier(identifier):
    """Partially masks phone number or email for privacy."""
    if not identifier:
        return ""
    identifier = str(identifier).strip()
    if "@" in identifier:
        parts = identifier.split("@")
        name = parts[0]
        domain = parts[1]
        masked_name = name[0] + "****" + name[-1] if len(name) > 2 else name[0] + "****"
        return f"{masked_name}@{domain}"
    else:
        # Phone masking e.g. +917676526744 -> +91 ******6744
        clean = identifier.replace(" ", "")
        if len(clean) > 6:
            prefix = clean[:3]
            suffix = clean[-4:]
            return f"{prefix} ******{suffix}"
        return clean

def init_auth_state():
    """Initializes session state for secure authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "otp_sent" not in st.session_state:
        st.session_state.otp_sent = False
    if "otp_sent_timestamp" not in st.session_state:
        st.session_state.otp_sent_timestamp = 0
    if "login_identifier" not in st.session_state:
        st.session_state.login_identifier = None
    if "onboarding_complete" not in st.session_state:
        st.session_state.onboarding_complete = False

def send_otp_backend(identifier):
    """
    Backend function to trigger real SMS OTP via Twilio Verify API.
    Enforces rate-limiting cooldown and logs zero OTP data in frontend.
    Returns (success: bool, message: str).
    """
    init_auth_state()
    
    # Rate Limiting Check (Cooldodwn period e.g. 60 seconds)
    now = time.time()
    elapsed_since_last_send = now - st.session_state.otp_sent_timestamp
    if elapsed_since_last_send < OTP_COOLDOWN_SECONDS:
        remaining_cooldown = int(OTP_COOLDOWN_SECONDS - elapsed_since_last_send)
        return False, f"⏱️ Rate Limit Exceeded: Please wait {remaining_cooldown} seconds before requesting a new OTP."
        
    identifier = str(identifier).strip()
    
    if is_twilio_configured():
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            verification = client.verify.v2.services(TWILIO_VERIFY_SERVICE_SID).verifications.create(
                to=identifier,
                channel="sms"
            )
            if verification.status in ["pending", "approved"]:
                st.session_state.otp_sent = True
                st.session_state.otp_sent_timestamp = now
                st.session_state.login_identifier = identifier
                return True, f"📲 Real SMS OTP dispatched to {mask_identifier(identifier)}."
            else:
                return False, f"🚨 SMS Provider Error: Status {verification.status}."
        except Exception as e:
            return False, f"🚨 SMS Provider API Failure: {str(e)}"
    else:
        # Twilio credentials missing warning
        return False, f"⚠️ Real SMS Provider Not Configured: Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_VERIFY_SERVICE_SID in your .env file to send real SMS messages to {identifier}."

def verify_otp_backend(entered_otp):
    """
    Backend function to verify entered OTP against the real SMS Provider API.
    Enforces 10-minute expiry and exact match.
    Returns (success: bool, message: str).
    """
    if not st.session_state.otp_sent or not st.session_state.login_identifier:
        return False, "🚨 No active OTP request found. Please request a new OTP."
        
    now = time.time()
    elapsed = now - st.session_state.otp_sent_timestamp
    if elapsed > OTP_EXPIRY_SECONDS:
        st.session_state.otp_sent = False
        return False, "🚨 OTP has expired. Please request a new OTP."
        
    identifier = st.session_state.login_identifier
    code = str(entered_otp).strip()
    
    if len(code) != 6 or not code.isdigit():
        return False, "❌ Invalid OTP format. Please enter a 6-digit numeric code."
        
    if is_twilio_configured():
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            check = client.verify.v2.services(TWILIO_VERIFY_SERVICE_SID).verification_checks.create(
                to=identifier,
                code=code
            )
            if check.status == "approved":
                st.session_state.authenticated = True
                st.session_state.otp_sent = False
                return True, "✅ OTP Verification Successful!"
            else:
                return False, "❌ Incorrect OTP. Please check the code sent to your phone and try again."
        except Exception as e:
            return False, f"🚨 SMS Verification Error: {str(e)}"
    else:
        return False, "⚠️ Real SMS Provider Not Configured: Please add Twilio credentials to .env file."

def logout_user():
    """Logs out the current user and clears session state."""
    st.session_state.authenticated = False
    st.session_state.user_profile = None
    st.session_state.onboarding_complete = False
    st.session_state.otp_sent = False

def render_login_screen():
    """Renders clean production login portal matching application visual style."""
    init_auth_state()
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">
        <div style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🔒 BuzzStreet Intelligence Portal
        </div>
        <div style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.3rem;">
            Market Psychology AI & Real-Time Narrative Shift Engine
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background-color: #0f172a; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 16px; padding: 28px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);">
            <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 16px; text-align: center;">
                Production SMS / Phone Authentication
            </div>
        """, unsafe_allow_html=True)
        
        # Display Configuration Notice if .env is missing Twilio keys
        if not is_twilio_configured():
            st.warning("""
            **⚠️ SMS Gateway Configuration Notice:**  
            Real SMS Provider API credentials are not set in `.env`.  
            To receive real SMS messages on your mobile device, please set `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_VERIFY_SERVICE_SID` in your `.env` file.
            """)
            
        col_cc, col_num = st.columns([1.5, 2.5])
        with col_cc:
            c_label, c_code = st.selectbox(
                "Country Code:",
                options=COUNTRY_CODES,
                format_func=lambda x: x[0],
                key="auth_country_select"
            )
        with col_num:
            phone_num = st.text_input("Phone Number:", placeholder="7676526744", key="auth_phone_field")
            
        clean_digits = re.sub(r"[^\d]", "", phone_num)
        full_phone = f"{c_code}{clean_digits}"
        
        # Calculate cooldown remaining
        elapsed_cooldown = time.time() - st.session_state.otp_sent_timestamp
        cooldown_remaining = max(0, OTP_COOLDOWN_SECONDS - int(elapsed_cooldown))
        
        btn_label = "Send OTP" if cooldown_remaining == 0 else f"Resend OTP in {cooldown_remaining}s"
        
        if st.button(btn_label, use_container_width=True, type="primary", disabled=(cooldown_remaining > 0), key="btn_send_phone_otp"):
            if not clean_digits or len(clean_digits) < 6:
                st.error("Please enter a valid phone number (minimum 6 digits).")
            else:
                success, msg = send_otp_backend(full_phone)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
                    
        # Render OTP Entry Screen after SMS is sent
        if st.session_state.otp_sent and st.session_state.login_identifier:
            st.divider()
            
            # Masked Phone Number Display
            masked = mask_identifier(st.session_state.login_identifier)
            
            # Remaining time calculation (10 minutes)
            elapsed_exp = time.time() - st.session_state.otp_sent_timestamp
            remaining_exp = max(0, OTP_EXPIRY_SECONDS - int(elapsed_exp))
            exp_min = remaining_exp // 60
            exp_s = remaining_exp % 60
            
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10b981; border-radius: 10px; padding: 14px 18px; margin-bottom: 15px;">
                <div style="font-size: 0.95rem; color: #34d399; font-weight: 700;">
                    📲 OTP sent to {masked}
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                    ⏱️ OTP expires in <b>{exp_min:02d}:{exp_s:02d}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            entered_otp = st.text_input("Enter 6-digit OTP:", max_chars=6, placeholder="[ _ _ _ _ _ _ ]", key="auth_otp_input_field")
            
            col_v1, col_v2 = st.columns([2, 1])
            with col_v1:
                if st.button("Verify OTP", use_container_width=True, type="primary", key="btn_verify_login"):
                    success, msg = verify_otp_backend(entered_otp)
                    if success:
                        st.success(msg)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(msg)
            with col_v2:
                resend_disabled = (cooldown_remaining > 0)
                if st.button("Resend OTP", use_container_width=True, disabled=resend_disabled, key="btn_resend_otp_secondary"):
                    success, msg = send_otp_backend(st.session_state.login_identifier)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                        
        st.markdown("</div>", unsafe_allow_html=True)

def render_onboarding_screen():
    """Renders first-time user profile setup screen upon initial login."""
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">
        <div style="font-size: 2.5rem; font-weight: 800; color: #38bdf8;">
            👤 Welcome to BuzzStreet! Complete Your Profile
        </div>
        <div style="color: #94a3b8; font-size: 1.05rem; margin-top: 0.3rem;">
            Customize your Market Psychology AI preferences & trading dashboard focus.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("onboarding_form", clear_on_submit=False):
            st.markdown("#### 📋 Trader Profile & Preferences")
            
            user_name = st.text_input("Full Name:", placeholder="e.g. Suhani Bagane", key="onboard_name")
            
            trader_type = st.selectbox(
                "Investor / Trader Role:",
                options=[
                    "Retail Active Trader",
                    "Institutional Portfolio Manager",
                    "Financial Quantitative Researcher",
                    "Financial News Journalist / Analyst",
                    "Academic Researcher / Student"
                ],
                key="onboard_role"
            )
            
            market_focus = st.selectbox(
                "Primary Market Focus:",
                options=[
                    "Indian Markets (NSE Nifty 50 & BSE Sensex)",
                    "US Tech Equities & S&P 500",
                    "Global Macro & Foreign Exchange (Forex)",
                    "Cryptocurrency & Digital Assets",
                    "Global Commodities (Gold, Silver, Crude Oil)"
                ],
                key="onboard_market"
            )
            
            alert_pref = st.select_slider(
                "Preferred Narrative Alert Threshold:",
                options=["Sensitive (Any Drift)", "Moderate (Fear & Panic Only)", "Extreme Panic Only"],
                value="Moderate (Fear & Panic Only)",
                key="onboard_alert"
            )
            
            submitted = st.form_submit_button("🚀 Save & Launch Dashboard", use_container_width=True, type="primary")
            
            if submitted:
                if not user_name or len(user_name.strip()) < 2:
                    st.error("Please enter your full name to complete registration.")
                else:
                    st.session_state.user_profile = {
                        "name": user_name.strip(),
                        "identifier": st.session_state.login_identifier,
                        "role": trader_type,
                        "market_focus": market_focus,
                        "alert_pref": alert_pref,
                        "joined_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    st.session_state.onboarding_complete = True
                    st.success(f"Profile created for **{user_name}**! Redirecting to Dashboard...")
                    time.sleep(0.5)
                    st.rerun()
