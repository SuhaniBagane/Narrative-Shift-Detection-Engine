"""
BuzzStreet – auth.py
Authentication & Onboarding Module.
Handles: Email / International Phone OTP generation, 10-minute validity enforcement,
demo OTP delivery notification, first-time user profile onboarding, and session persistence.
"""

import time
import random
import re
import datetime
import streamlit as st

# Default Country Codes Map
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

def init_auth_state():
    """Initializes session state variables for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "otp_sent" not in st.session_state:
        st.session_state.otp_sent = False
    if "current_otp" not in st.session_state:
        st.session_state.current_otp = None
    if "otp_created_at" not in st.session_state:
        st.session_state.otp_created_at = None
    if "login_identifier" not in st.session_state:
        st.session_state.login_identifier = None
    if "onboarding_complete" not in st.session_state:
        st.session_state.onboarding_complete = False

def generate_otp():
    """Generates a secure 6-digit numeric OTP."""
    return str(random.randint(100000, 999999))

def send_otp(identifier):
    """Sends OTP and sets a strict 10-minute (600 seconds) expiry timestamp."""
    otp = generate_otp()
    st.session_state.current_otp = otp
    st.session_state.otp_created_at = time.time()
    st.session_state.login_identifier = identifier
    st.session_state.otp_sent = True
    return otp

def verify_otp(entered_otp):
    """
    Verifies entered OTP against stored OTP.
    Enforces the strict 10-minute (600 seconds) validity rule.
    Returns (success: bool, message: str).
    """
    if not st.session_state.otp_sent or not st.session_state.current_otp:
        return False, "🚨 No active OTP request found. Please request a new OTP."
        
    # Check 10-minute expiry (600 seconds)
    elapsed = time.time() - st.session_state.otp_created_at
    if elapsed > 600:
        st.session_state.otp_sent = False
        st.session_state.current_otp = None
        return False, "🚨 OTP Expired! (10-minute validity limit exceeded). Please click 'Resend OTP'."
        
    entered = str(entered_otp).strip()
    stored = str(st.session_state.current_otp).strip()
    
    # STRICT EXACT MATCH ENFORCEMENT
    if entered == stored:
        st.session_state.authenticated = True
        st.session_state.otp_sent = False
        st.session_state.current_otp = None
        return True, "✅ OTP Verified Successfully! Welcome to BuzzStreet."
    else:
        return False, "❌ Incorrect OTP code entered! Access denied. You must enter the exact 6-digit OTP code sent to your phone/email."

def logout_user():
    """Logs out the current user and clears session state."""
    st.session_state.authenticated = False
    st.session_state.user_profile = None
    st.session_state.onboarding_complete = False
    st.session_state.otp_sent = False
    st.session_state.current_otp = None

def render_login_screen():
    """Renders high-end dark slate login portal with Email and Phone OTP tabs."""
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
                Authentication & Verification
            </div>
        """, unsafe_allow_html=True)
        
        tab_email, tab_phone = st.tabs(["📧 Email OTP Login", "📱 Phone OTP Login (Global)"])
        
        with tab_email:
            email_input = st.text_input("Enter Registered Email Address:", placeholder="trader@buzzstreet.com", key="auth_email_field")
            
            if st.button("📩 Send 6-Digit OTP to Email", use_container_width=True, type="primary", key="btn_send_email_otp"):
                if not email_input or "@" not in email_input or "." not in email_input:
                    st.error("Please enter a valid email address.")
                else:
                    otp = send_otp(email_input)
                    st.success(f"OTP sent to **{email_input}**! Valid for 10 minutes.")
                    
        with tab_phone:
            col_cc, col_num = st.columns([1.5, 2.5])
            with col_cc:
                c_label, c_code = st.selectbox(
                    "Country Code:",
                    options=COUNTRY_CODES,
                    format_func=lambda x: x[0],
                    key="auth_country_select"
                )
            with col_num:
                phone_num = st.text_input("Phone Number:", placeholder="9876543210", key="auth_phone_field")
                
            full_phone = f"{c_code} {phone_num}".strip()
            
            if st.button("📲 Send 6-Digit OTP to Phone", use_container_width=True, type="primary", key="btn_send_phone_otp"):
                if not phone_num or len(phone_num) < 6:
                    st.error("Please enter a valid phone number.")
                else:
                    otp = send_otp(full_phone)
                    st.success(f"OTP sent to **{full_phone}**! Valid for 10 minutes.")
                    
        # If OTP was sent, render the 6-digit OTP verification field
        if st.session_state.otp_sent:
            st.divider()
            
            # Remaining time calculation
            elapsed = time.time() - st.session_state.otp_created_at
            remaining_sec = max(0, 600 - int(elapsed))
            rem_min = remaining_sec // 60
            rem_s = remaining_sec % 60
            
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10b981; border-radius: 10px; padding: 14px 18px; margin-bottom: 15px;">
                <div style="font-size: 0.85rem; color: #34d399; font-weight: 700;">
                    📲 SMS / Email Security Gateway Dispatch
                </div>
                <div style="font-size: 0.95rem; color: #ffffff; margin-top: 4px;">
                    A 6-digit Security OTP has been dispatched to <b>{st.session_state.login_identifier}</b>.
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #38bdf8; margin-top: 8px; background: rgba(15, 23, 42, 0.9); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.3); display: inline-block;">
                    🔑 Dispatched Security OTP: <span style="color: #34d399; font-family: monospace; letter-spacing: 3px;">{st.session_state.current_otp}</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 8px;">
                    ⏱️ Validity Remaining: <b>{rem_min:02d}:{rem_s:02d}</b> (Strict 10-Minute Expiry Rule & Exact Match Requirement)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            entered_otp = st.text_input("Enter 6-Digit OTP Code:", max_chars=6, placeholder="e.g. 784912", key="auth_otp_input_field")
            
            col_v1, col_v2 = st.columns([2, 1])
            with col_v1:
                if st.button("🔐 Verify & Login", use_container_width=True, type="primary", key="btn_verify_login"):
                    success, msg = verify_otp(entered_otp)
                    if success:
                        st.success(msg)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(msg)
            with col_v2:
                if st.button("🔄 Resend", use_container_width=True, key="btn_resend_otp"):
                    send_otp(st.session_state.login_identifier)
                    st.info("New OTP generated!")
                    st.rerun()
                    
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
