from __future__ import annotations

from datetime import datetime

import streamlit as st

from database import authenticate_user, create_user, log_action
from styles import render_login_intro


ROLES = ["Analyst", "Manager", "Admin"]


def ensure_session() -> None:
    defaults = {
        "logged_in": False,
        "current_user": None,
        "user_role": None,
        "full_name": None,
        "login_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def login_page() -> None:
    render_login_intro()
    tab1, tab2 = st.tabs(["🔐 Login", "✨ Sign Up"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.current_user = user["username"]
                    st.session_state.user_role = user["role"]
                    st.session_state.full_name = user["full_name"]
                    st.session_state.login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_action(user["username"], user["role"], "login_success", "User logged in")
                    st.success("Login successful")
                    st.rerun()
                else:
                    log_action(username or "unknown", None, "login_failed", "Invalid credentials")
                    st.error("Invalid username or password")
    with tab2:
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            username = st.text_input("New Username")
            password = st.text_input("New Password", type="password")
            role = st.selectbox("Role", ROLES, index=0)
            submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                try:
                    create_user(username, password, role, full_name)
                    log_action(username, role, "user_created", f"Created with role {role}")
                    st.success("Account created. Log in on the Login tab.")
                except Exception as exc:
                    st.error(f"Unable to create account: {exc}")


def logout() -> None:
    log_action(st.session_state.current_user, st.session_state.user_role, "logout", "User logged out")
    for key in ["logged_in", "current_user", "user_role", "full_name", "login_time"]:
        st.session_state[key] = None if key != "logged_in" else False
    st.rerun()


def require_login() -> None:
    if not st.session_state.get("logged_in"):
        st.warning("Please log in.")
        st.stop()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 👤 Profile")
        st.markdown(
            f"""
            <div class="profile-card">
                <div style="font-weight:700; font-size:1rem;">{st.session_state.get("full_name") or st.session_state.get("current_user")}</div>
                <div class="small-note">Role: {st.session_state.get("user_role","-")}</div>
                <div class="small-note">Login: {st.session_state.get("login_time","-")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("🚪 Logout", on_click=logout, use_container_width=True)
        st.info("Model training is cached for speed. Use the horizontal menu above to move between engines.")
