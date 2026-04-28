from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root{
            --bg0:#f8e9f2; --bg1:#f6ddea; --bg2:#f9efe5;
            --card:#fffafb; --card2:#fff6f8; --text:#241428; --muted:#5f4f63;
            --accent:#7c3aed; --accent2:#f97316; --accent3:#ec4899;
        }

        .stApp{
            background:
                radial-gradient(circle at 10% 15%, rgba(244,114,182,0.10), transparent 24%),
                radial-gradient(circle at 88% 12%, rgba(251,146,60,0.08), transparent 26%),
                radial-gradient(circle at 80% 82%, rgba(124,58,237,0.08), transparent 24%),
                linear-gradient(135deg, var(--bg0), var(--bg1) 50%, var(--bg2));
        }

        [data-testid="stHeader"]{background:transparent;}
        [data-testid="stSidebar"]{
            background:linear-gradient(180deg, rgba(116,70,122,0.92), rgba(145,91,103,0.90));
            border-right:1px solid rgba(255,255,255,0.22);
        }
        [data-testid="stSidebarNav"]{display:none;}

        .hero-card,.panel-card,.profile-card,.feature-card,.login-card{
            background:linear-gradient(180deg, rgba(255,247,251,0.98), rgba(255,242,247,0.98));
            border:1px solid rgba(255,255,255,0.50);
            box-shadow:0 24px 60px rgba(46,11,40,0.18);
            border-radius:26px;
            color:var(--text);
        }

        .hero-card{padding:1.45rem 1.55rem; margin-bottom:1rem;}
        .panel-card{padding:1rem 1.1rem; margin-bottom:1rem;}
        .profile-card{padding:1rem 1rem; margin-top:.4rem;}
        .feature-card{padding:1rem 1rem; min-height:170px; transition: transform .25s ease, box-shadow .25s ease;}
        .feature-card:hover{transform:translateY(-5px); box-shadow:0 26px 64px rgba(46,11,40,0.24);}
        .top-nav-wrap{margin-bottom:.35rem;}

        .brand-pill{
            display:inline-block; padding:.45rem .8rem; border-radius:999px;
            background:linear-gradient(135deg, rgba(124,58,237,0.14), rgba(249,115,22,0.16));
            color:#5b21b6; font-weight:800; font-size:.78rem; letter-spacing:.12rem; text-transform:uppercase;
        }
        .section-title{
            font-size:2.2rem; font-weight:800; margin:.55rem 0 .4rem 0;
            color:#221126; font-family:"Trebuchet MS","Segoe UI",sans-serif;
        }
        .section-copy{color:#4f4453; line-height:1.65; font-size:1rem;}
        .small-note{color:#54475a; font-size:.89rem;}

        .metric-banner{
            border-radius:20px; padding:1rem 1rem .95rem 1rem;
            background:linear-gradient(160deg, rgba(255,252,253,.99), rgba(255,244,247,.99));
            border:1px solid rgba(124,58,237,0.12);
            box-shadow:0 14px 36px rgba(46,11,40,0.12);
            min-height:124px;
            position:relative; overflow:hidden;
        }
        .metric-banner:before{
            content:""; position:absolute; inset:0 auto auto 0; width:100%; height:4px;
            background:linear-gradient(90deg, #7c3aed, #ec4899, #f97316);
        }
        .metric-banner .label{font-size:.76rem; text-transform:uppercase; letter-spacing:.12rem; color:#6e5874; font-weight:700;}
        .metric-banner .value{font-size:1.88rem; font-weight:800; margin:.42rem 0 .2rem 0; color:#2f1736;}
        .metric-banner .caption{font-size:.90rem; color:#5a4b5f;}

        .pulse-logo{
            width:94px;height:94px;border-radius:30px;display:grid;place-items:center;
            background:linear-gradient(135deg, rgba(124,58,237,0.10), rgba(249,115,22,0.14));
            border:1px solid rgba(124,58,237,0.12); margin:1rem 0 1rem 0; animation:pulse 2.6s infinite;
        }
        .pulse-logo span{font-size:2rem}
        @keyframes pulse{
            0%{box-shadow:0 0 0 0 rgba(236,72,153,0.18)}
            70%{box-shadow:0 0 0 18px rgba(236,72,153,0)}
            100%{box-shadow:0 0 0 0 rgba(236,72,153,0)}
        }

        .stTabs [data-baseweb="tab"]{
            border-radius:12px; background:rgba(124,58,237,0.08); color:#37153f;
            border:1px solid rgba(124,58,237,0.12);
        }
        .stTabs [aria-selected="true"]{
            background:linear-gradient(135deg, rgba(124,58,237,0.95), rgba(249,115,22,0.85));
            color:white;
        }
        .stButton>button,.stDownloadButton>button{
            border-radius:14px; border:1px solid rgba(124,58,237,.12);
            background:linear-gradient(135deg, #7c3aed, #f97316);
            color:white; font-weight:800;
        }
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div{
            border-radius:14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, eyebrow: str = "POSB Credit Studio") -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="brand-pill">{eyebrow}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_banner(label: str, value: str, caption: str) -> str:
    return f"""
    <div class="metric-banner">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="caption">{caption}</div>
    </div>
    """


def feature_card(title: str, body: str) -> str:
    return f"""
    <div class="feature-card">
        <div style="font-weight:800; font-size:1.08rem; margin-bottom:.4rem; color:#2f1736;">{title}</div>
        <div class="section-copy" style="font-size:.95rem;">{body}</div>
    </div>
    """


def render_login_intro() -> None:
    st.markdown(
        """
        <div class="login-card" style="padding:1.9rem 1.9rem; margin-top:3vh;">
            <div class="brand-pill">Credit Studio</div>
            <div class="pulse-logo"><span>◆</span></div>
            <div class="section-title" style="font-size:2.45rem;">Twin-Stage Bayesian–XGBoost Studio</div>
            <div class="section-copy">
                A premium POSB credit scoring workspace for posterior-updated analytics, nonlinear stacking, forecasting, and governance-ready credit decisions.
            </div>
            <div style="margin-top:1rem;" class="small-note">
                Designed with a warm-spectrum capital-markets aesthetic to feel materially different from prior IFRS 9 tooling.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
