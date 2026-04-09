import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional lightweight dependencies (guarded)
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_READER_AVAILABLE = True
except Exception:
    PDF_READER_AVAILABLE = False


# ============================================================
# Context & Goal
# ------------------------------------------------------------
# This app is designed for employment center counselors to quickly analyze
# hiring trends and generate practical consulting insights from job postings.
# The goal is to reduce information asymmetry with reliable, demo-safe UX.
# ============================================================

st.set_page_config(
    page_title="온톱(On-Top) 채용 트렌드 인사이트",
    page_icon="📈",
    layout="wide"
)

CORE_SKILL_POOL = [
    "Python", "SQL", "Java", "C", "C++", "JavaScript", "TypeScript", "React", "Node.js",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Linux", "Git",
    "TensorFlow", "PyTorch", "LLM", "RAG", "MLOps", "NLP",
    "데이터분석", "통계", "문제해결", "협업", "커뮤니케이션",
    "반도체", "회로설계", "공정", "검증", "FPGA", "Verilog",
]

CERT_POOL = ["정보처리기사", "SQLD", "ADsP", "빅데이터분석기사", "AWS SAA", "PMP"]

JOB_KEYWORDS = {
    "데이터": ["데이터", "Data", "분석", "AI", "ML", "머신러닝", "LLM"],
    "백엔드": ["백엔드", "서버", "Backend", "API", "플랫폼"],
    "프론트엔드": ["프론트", "Frontend", "UI", "웹개발"],
    "반도체": ["반도체", "DS", "공정", "회로", "소자", "검증"],
    "영업": ["영업", "세일즈", "고객", "마케팅"],
    "기획": ["기획", "PM", "전략", "서비스"],
}


# -------------------------------
# [1] Data extraction & cleansing
# -------------------------------
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize various input schemas into one robust schema."""
    df = df.copy()

    rename_map = {
        "date": "posting_date",
        "공고일": "posting_date",
        "채용공고일": "posting_date",
        "year": "year",
        "연도": "year",
        "half": "half",
        "반기": "half",
        "company": "company",
        "기업": "company",
        "company_group": "company_group",
        "그룹사": "company_group",
        "business_unit": "business_unit",
        "사업부": "business_unit",
        "industry": "industry",
        "산업군": "industry",
        "job_category": "job_category",
        "직무카테고리": "job_category",
        "job_type": "job_category",
        "job_title": "job_title",
        "직무": "job_title",
        "required_skills": "required_skills",
        "필수역량": "required_skills",
        "preferred_skills": "preferred_skills",
        "우대역량": "preferred_skills",
        "preferred_certifications": "preferred_certifications",
        "자격증": "preferred_certifications",
        "preferred_education": "preferred_education",
        "학력": "preferred_education",
        "min_experience_years": "min_experience_years",
        "경력": "min_experience_years",
        "preferred_text": "preferred_text",
        "우대사항": "preferred_text",
    }

    matched = {}
    for c in df.columns:
        key = c.strip()
        if key in rename_map:
            matched[c] = rename_map[key]
    if matched:
        df = df.rename(columns=matched)

    return df


def generate_demo_data(n_rows: int = 900) -> pd.DataFrame:
    """Generate synthetic demo data for hackathon fallback."""
    rng = np.random.default_rng(42)

    years = [2021, 2022, 2023, 2024, 2025, 2026]
    halves = ["상반기", "하반기"]
    industries = ["IT", "제조", "금융", "바이오", "물류", "유통", "공공"]
    business_units = ["플랫폼", "데이터", "영업", "운영", "R&D", "HR", "마케팅"]
    job_categories = ["백엔드", "프론트엔드", "데이터", "기획", "영업", "운영", "디자인"]
    job_titles = {
        "백엔드": ["Backend Engineer", "Server Developer", "API Engineer"],
        "프론트엔드": ["Frontend Engineer", "UI Engineer", "Web Developer"],
        "데이터": ["Data Analyst", "Data Engineer", "ML Engineer"],
        "기획": ["Product Manager", "Service Planner"],
        "영업": ["B2B Sales", "Account Executive"],
        "운영": ["Operations Manager", "CS Lead"],
        "디자인": ["Product Designer", "UX Designer"],
    }
    companies = ["온톱테크", "미래솔루션", "넥스트잡", "그린모빌리티", "에이아이허브", "코어인더스트리"]
    groups = ["A그룹", "B그룹", "C그룹", "독립기업"]

    base_required = ["Python", "SQL", "커뮤니케이션", "문제해결", "협업", "데이터분석", "Excel"]
    trend_new_by_year = {
        2024: ["MLOps", "LLM"],
        2025: ["RAG", "Prompt Engineering"],
        2026: ["Agentic AI", "Synthetic Data"],
    }
    preferred_pool = ["AWS", "GCP", "Docker", "Kubernetes", "Tableau", "Power BI", "Git", "Notion"]
    cert_pool = ["정보처리기사", "SQLD", "ADsP", "AWS SAA", "빅데이터분석기사"]
    edu_pool = ["무관", "전문학사", "학사", "석사"]

    rows = []
    for _ in range(n_rows):
        year = int(rng.choice(years, p=[0.10, 0.12, 0.15, 0.18, 0.22, 0.23]))
        half = str(rng.choice(halves, p=[0.48, 0.52]))
        month = int(rng.choice([2, 3, 4, 5, 9, 10, 11, 12]))
        day = int(rng.integers(1, 28))

        industry = str(rng.choice(industries))
        job_category = str(rng.choice(job_categories))
        job_title = str(rng.choice(job_titles[job_category]))

        required = set(rng.choice(base_required, size=int(rng.integers(3, 6)), replace=False).tolist())
        for y, new_skills in trend_new_by_year.items():
            if year >= y and rng.random() < 0.35:
                required.add(str(rng.choice(new_skills)))

        preferred = set(rng.choice(preferred_pool, size=int(rng.integers(2, 5)), replace=False).tolist())
        cert = str(rng.choice(cert_pool)) if rng.random() < 0.35 else ""
        edu = str(rng.choice(edu_pool, p=[0.30, 0.15, 0.45, 0.10]))
        exp = int(rng.choice([0, 1, 2, 3, 5, 7], p=[0.12, 0.16, 0.28, 0.22, 0.16, 0.06]))

        ai_bias = 0.12 + 0.06 * (year - 2021)
        preferred_text = ""
        if rng.random() < min(0.60, ai_bias):
            preferred_text = "AI·데이터 프로젝트 경험 우대"

        rows.append(
            {
                "posting_date": datetime(year, month, day).strftime("%Y-%m-%d"),
                "year": year,
                "half": half,
                "company": str(rng.choice(companies)),
                "company_group": str(rng.choice(groups)),
                "business_unit": str(rng.choice(business_units)),
                "industry": industry,
                "job_category": job_category,
                "job_title": job_title,
                "required_skills": ", ".join(sorted(required)),
                "preferred_skills": ", ".join(sorted(preferred)),
                "preferred_certifications": cert,
                "preferred_education": edu,
                "min_experience_years": exp,
                "preferred_text": preferred_text,
            }
        )

    return pd.DataFrame(rows)


def safe_parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "posting_date" in df.columns:
        try:
            df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
        except Exception:
            df["posting_date"] = pd.NaT

        if "year" not in df.columns:
            df["year"] = df["posting_date"].dt.year
        if "half" not in df.columns:
            month = df["posting_date"].dt.month
            df["half"] = np.where(month <= 6, "상반기", "하반기")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


def split_skills(skill_text: str):
    if pd.isna(skill_text):
        return []
    text = str(skill_text).strip()
    if not text:
        return []
    parts = re.split(r"[,/|;\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def explode_skills(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["year", "job_category", "industry", "skill"]) 

    temp = df[[c for c in ["year", "job_category", "industry", col] if c in df.columns]].copy()
    temp["skill"] = temp[col].apply(split_skills)
    temp = temp.explode("skill")
    temp = temp.dropna(subset=["skill"])
    temp["skill"] = temp["skill"].astype(str).str.strip()
    temp = temp[temp["skill"] != ""]
    return temp


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing columns with safe defaults to prevent red-screen in demo."""
    df = df.copy()
    defaults = {
        "posting_date": pd.NaT,
        "year": pd.NA,
        "half": "미상",
        "company": "미상",
        "company_group": "미상",
        "business_unit": "미상",
        "industry": "미상",
        "job_category": "미상",
        "job_title": "미상",
        "required_skills": "",
        "preferred_skills": "",
        "preferred_certifications": "",
        "preferred_education": "무관",
        "min_experience_years": 0,
        "preferred_text": "",
        "source_pdf": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["min_experience_years"] = pd.to_numeric(df["min_experience_years"], errors="coerce").fillna(0)
    return df


def find_pdf_files(data_dir: str = "data") -> List[Path]:
    """Return all PDF files under data directory with deterministic ordering."""
    try:
        root = Path(data_dir)
        if not root.exists():
            return []
        return sorted(root.rglob("*.pdf"))
    except Exception:
        return []


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 20) -> str:
    """Read text safely from PDF. Returns empty string when extraction fails."""
    if not PDF_READER_AVAILABLE:
        return ""

    try:
        reader = PdfReader(str(pdf_path))
        pages = reader.pages[:max_pages]
        chunks = []
        for page in pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text)
        return "\n".join(chunks)
    except Exception:
        return ""


def parse_year_half(file_name: str, text: str) -> Tuple[Optional[int], str]:
    raw = f"{file_name}\n{text[:1200]}"
    year = None
    half = "미상"

    m4 = re.search(r"(20\d{2})", raw)
    if m4:
        year = int(m4.group(1))
    else:
        m2 = re.search(r"\b(\d{2})\s*(상|하)\b", raw)
        if m2:
            yy = int(m2.group(1))
            year = 2000 + yy

    if re.search(r"(상반기|상\s*반기|상\b|1H|H1|상\s*채)", raw, flags=re.IGNORECASE):
        half = "상반기"
    if re.search(r"(하반기|하\s*반기|하\b|2H|H2|하\s*채)", raw, flags=re.IGNORECASE):
        half = "하반기"

    if half == "미상" and year is not None:
        half = "상반기"
    return year, half


def infer_metadata_from_path(pdf_path: Path) -> dict:
    path_low = str(pdf_path).lower()
    company = "미상"
    company_group = "독립기업"
    business_unit = "미상"
    industry = "미상"

    if "samsung" in path_low or "삼성" in path_low:
        company = "삼성전자"
        company_group = "삼성"

    if " ds" in path_low or "\\ds\\" in path_low or "/ds/" in path_low:
        business_unit = "DS"
        industry = "반도체/전자"
    elif " dx" in path_low or "\\dx\\" in path_low or "/dx/" in path_low:
        business_unit = "DX"
        industry = "IT/디지털전환"

    return {
        "company": company,
        "company_group": company_group,
        "business_unit": business_unit,
        "industry": industry,
    }


def guess_job_category(text: str, business_unit: str) -> str:
    text_l = text.lower()
    counts = {}
    for cat, keys in JOB_KEYWORDS.items():
        score = sum(1 for k in keys if k.lower() in text_l)
        if score > 0:
            counts[cat] = score
    if counts:
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
    if business_unit == "DS":
        return "반도체"
    if business_unit == "DX":
        return "데이터"
    return "미상"


def extract_skills_from_text(text: str, business_unit: str, top_n: int = 8) -> List[str]:
    found = []
    text_l = text.lower()

    for skill in CORE_SKILL_POOL:
        if re.fullmatch(r"[A-Za-z0-9.+#\-]+", skill):
            if re.search(rf"(?<!\w){re.escape(skill.lower())}(?!\w)", text_l):
                found.append(skill)
        else:
            if skill in text:
                found.append(skill)

    if not found:
        if business_unit == "DS":
            found = ["반도체", "회로설계", "공정", "문제해결", "협업"]
        elif business_unit == "DX":
            found = ["Python", "SQL", "데이터분석", "문제해결", "협업"]
        else:
            found = ["문제해결", "협업", "커뮤니케이션"]
    return found[:top_n]


def extract_preferred_text(text: str, max_len: int = 180) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hit = [ln for ln in lines if ("우대" in ln or "prefer" in ln.lower())]
    if not hit:
        return ""
    merged = " / ".join(hit[:3]).strip()
    return merged[:max_len]


def extract_education(text: str) -> str:
    if "박사" in text:
        return "박사"
    if "석사" in text:
        return "석사"
    if "학사" in text:
        return "학사"
    if "전문학사" in text:
        return "전문학사"
    if "무관" in text:
        return "무관"
    return "무관"


def extract_experience_years(text: str) -> int:
    try:
        matches = re.findall(r"(\d+)\s*년", text)
        if not matches:
            return 0
        years = [int(x) for x in matches if x.isdigit()]
        if not years:
            return 0
        return int(min(max(years), 15))
    except Exception:
        return 0


def extract_certifications(text: str) -> str:
    certs = [c for c in CERT_POOL if c in text]
    return ", ".join(certs[:3]) if certs else ""


def parse_pdf_to_record(pdf_path: Path) -> dict:
    meta = infer_metadata_from_path(pdf_path)
    raw_text = extract_text_from_pdf(pdf_path)
    year, half = parse_year_half(pdf_path.name, raw_text)

    if year is None:
        year = datetime.now().year
    month = 3 if half == "상반기" else 9

    job_category = guess_job_category(raw_text, meta["business_unit"])
    required = extract_skills_from_text(raw_text, meta["business_unit"], top_n=8)
    preferred = extract_skills_from_text(raw_text, meta["business_unit"], top_n=5)
    pref_text = extract_preferred_text(raw_text)
    edu = extract_education(raw_text)
    exp = extract_experience_years(raw_text)
    cert = extract_certifications(raw_text)

    return {
        "posting_date": f"{year}-{month:02d}-01",
        "year": year,
        "half": half,
        "company": meta["company"],
        "company_group": meta["company_group"],
        "business_unit": meta["business_unit"],
        "industry": meta["industry"],
        "job_category": job_category,
        "job_title": pdf_path.stem,
        "required_skills": ", ".join(required),
        "preferred_skills": ", ".join(preferred),
        "preferred_certifications": cert,
        "preferred_education": edu,
        "min_experience_years": exp,
        "preferred_text": pref_text,
        "source_pdf": str(pdf_path),
    }


def load_pdf_dataset(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build dataset from PDF files in data folder.
    Returns:
        df: parsed records
        ingest_report: per-file parsing status
    """
    files = find_pdf_files(data_dir)
    if not files:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    status_rows = []
    for pdf in files:
        try:
            record = parse_pdf_to_record(pdf)
            rows.append(record)
            status_rows.append(
                {
                    "file": pdf.name,
                    "status": "ok",
                    "year": record["year"],
                    "half": record["half"],
                    "business_unit": record["business_unit"],
                }
            )
        except Exception as e:
            status_rows.append(
                {
                    "file": pdf.name,
                    "status": f"error: {e}",
                    "year": None,
                    "half": "미상",
                    "business_unit": "미상",
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(status_rows)


# ----------------------------------------
# [2] AI-ready trend logic (without hard key)
# ----------------------------------------
def suggest_top5_skills(required_skill_exploded: pd.DataFrame, selected_job: str) -> pd.DataFrame:
    try:
        temp = required_skill_exploded.copy()
        if temp.empty:
            return pd.DataFrame(columns=["skill", "score"])

        if selected_job != "전체":
            temp = temp[temp["job_category"] == selected_job]

        recent_years = sorted([y for y in temp["year"].dropna().unique().tolist() if pd.notna(y)])
        if not recent_years:
            return pd.DataFrame(columns=["skill", "score"])

        latest_3 = recent_years[-3:]
        temp = temp[temp["year"].isin(latest_3)]
        if temp.empty:
            return pd.DataFrame(columns=["skill", "score"])

        count_table = temp.groupby(["year", "skill"]).size().reset_index(name="cnt")

        year_weight = {y: (i + 1) for i, y in enumerate(latest_3)}
        count_table["weight"] = count_table["year"].map(year_weight).fillna(1)
        count_table["score"] = count_table["cnt"] * count_table["weight"]

        agg = count_table.groupby("skill", as_index=False)["score"].sum()
        agg = agg.sort_values("score", ascending=False).head(5)
        return agg
    except Exception:
        return pd.DataFrame(columns=["skill", "score"])


def forecast_2026_postings(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Simple linear forecast for 2026 hiring outlook."""
    try:
        yearly = df.dropna(subset=["year"]).groupby("year").size().reset_index(name="postings")
        yearly = yearly.sort_values("year")
        yearly = yearly[yearly["year"] >= 2021]

        if len(yearly) < 3:
            return "데이터가 3개 연도 미만이라 2026 전망 신뢰도가 낮습니다.", yearly

        x = yearly["year"].astype(float).values
        y = yearly["postings"].astype(float).values

        coeff = np.polyfit(x, y, deg=1)
        pred_2026 = int(max(0, round(np.polyval(coeff, 2026))))
        latest_year = int(yearly["year"].max())
        latest_cnt = int(yearly.loc[yearly["year"] == latest_year, "postings"].iloc[0])

        direction = "증가" if pred_2026 >= latest_cnt else "감소"
        msg = f"2026년 예상 공고 수는 약 {pred_2026}건으로 추정되며, 직전 연도({latest_year}) 대비 {direction} 흐름입니다."
        return msg, yearly
    except Exception:
        return "전망 계산 중 오류가 발생해 정성 인사이트만 제공합니다.", pd.DataFrame()


# -------------------------------
# [3] Streamlit UI + review step
# -------------------------------
def sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("필터")

    years = sorted([int(y) for y in df["year"].dropna().unique().tolist()])
    if years:
        selected_years = st.sidebar.multiselect("연도", years, default=years)
    else:
        selected_years = []

    industries = sorted(df["industry"].dropna().astype(str).unique().tolist())
    selected_industries = st.sidebar.multiselect("산업군", industries, default=industries)

    jobs = sorted(df["job_category"].dropna().astype(str).unique().tolist())
    selected_jobs = st.sidebar.multiselect("직무 카테고리", jobs, default=jobs)

    return selected_years, selected_industries, selected_jobs


def apply_filters(df: pd.DataFrame, years, industries, jobs) -> pd.DataFrame:
    temp = df.copy()
    if years:
        temp = temp[temp["year"].isin(years)]
    if industries:
        temp = temp[temp["industry"].isin(industries)]
    if jobs:
        temp = temp[temp["job_category"].isin(jobs)]
    return temp


def render_overview(df: pd.DataFrame):
    st.subheader("채용 트렌드 Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("공고 수", f"{len(df):,}")
    c2.metric("기업 수", f"{df['company'].nunique():,}")
    c3.metric("직무 종류 수", f"{df['job_category'].nunique():,}")

    # 연도별 채용 공고 수 변화
    yearly = df.dropna(subset=["year"]).groupby("year").size().reset_index(name="count")
    if not yearly.empty:
        fig = px.bar(yearly, x="year", y="count", text="count", title="연도별 채용 공고 수")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("연도 정보가 없어 연도별 공고 그래프를 표시할 수 없습니다.")

    # 산업군별 채용 비중 추이
    industry_year = df.dropna(subset=["year"]).groupby(["year", "industry"]).size().reset_index(name="count")
    if not industry_year.empty:
        industry_year["share"] = industry_year.groupby("year")["count"].transform(lambda x: x / max(x.sum(), 1))
        fig2 = px.area(industry_year, x="year", y="share", color="industry", groupnorm="fraction", title="산업군별 채용 비중 추이")
        fig2.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    # 연도별 상/하반기 사업부별 채용 추이
    by_half = df.dropna(subset=["year"]).groupby(["year", "half", "business_unit"]).size().reset_index(name="count")
    if not by_half.empty:
        fig3 = px.bar(by_half, x="year", y="count", color="business_unit", facet_col="half", barmode="stack",
                      title="연도별 상/하반기 사업부별 채용 추이")
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

    # 직무 종류 변화 추이
    job_year = df.dropna(subset=["year"]).groupby(["year", "job_category"]).size().reset_index(name="count")
    if not job_year.empty:
        fig4 = px.line(job_year, x="year", y="count", color="job_category", markers=True, title="직무 종류 변화 추이")
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

    # 상/하반기 채용 시즌 패턴
    season = df.groupby(["half", "industry"]).size().reset_index(name="count")
    if not season.empty:
        fig5 = px.bar(season, x="half", y="count", color="industry", barmode="group", title="상/하반기 채용 시즌 패턴")
        fig5.update_layout(height=360)
        st.plotly_chart(fig5, use_container_width=True)


def render_skills_analysis(df: pd.DataFrame):
    st.subheader("직무별 요구 역량 분석")

    req_exp = explode_skills(df, "required_skills")

    # 전체 공고 Top 키워드 (워드클라우드)
    st.markdown("**전체 공고 Top 키워드 (워드클라우드)**")
    skill_counter = Counter(req_exp["skill"].tolist()) if not req_exp.empty else Counter()

    if skill_counter:
        if WORDCLOUD_AVAILABLE:
            try:
                wc = WordCloud(width=1200, height=450, background_color="white", colormap="viridis")
                img = wc.generate_from_frequencies(dict(skill_counter))
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.imshow(img, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except Exception:
                top = pd.DataFrame(skill_counter.most_common(20), columns=["skill", "count"])
                st.info("워드클라우드 렌더링에 실패해 막대그래프로 대체합니다.")
                st.plotly_chart(px.bar(top, x="count", y="skill", orientation="h", title="Top 키워드"), use_container_width=True)
        else:
            top = pd.DataFrame(skill_counter.most_common(20), columns=["skill", "count"])
            st.info("`wordcloud` 라이브러리가 없어 막대그래프로 대체합니다.")
            st.plotly_chart(px.bar(top, x="count", y="skill", orientation="h", title="Top 키워드"), use_container_width=True)
    else:
        st.warning("필수 스킬 데이터가 비어 있어 키워드 분석을 표시할 수 없습니다.")

    col1, col2 = st.columns(2)

    # 3년간 꾸준히 요구된 필수 스킬
    with col1:
        st.markdown("**3년간 꾸준히 요구된 필수 스킬**")
        try:
            if not req_exp.empty:
                years = sorted([int(y) for y in req_exp["year"].dropna().unique().tolist()])
                last3 = years[-3:] if len(years) >= 3 else years
                stable = req_exp[req_exp["year"].isin(last3)].groupby(["skill", "year"]).size().reset_index(name="cnt")
                stable_count = stable.groupby("skill")["year"].nunique().reset_index(name="years_present")
                stable_skills = stable_count[stable_count["years_present"] == len(last3)].sort_values("years_present", ascending=False)
                st.dataframe(stable_skills.head(20), use_container_width=True)
            else:
                st.info("분석할 스킬 데이터가 없습니다.")
        except Exception:
            st.warning("꾸준히 요구된 스킬 계산 중 오류가 발생했습니다.")

    # 연도별 신규 등장 / 사라진 스킬
    with col2:
        st.markdown("**연도별 신규 등장 / 사라진 스킬**")
        try:
            if not req_exp.empty:
                yearly_skill = req_exp.groupby("year")["skill"].apply(set).to_dict()
                rows = []
                ys = sorted(yearly_skill.keys())
                for idx in range(1, len(ys)):
                    prev_y = ys[idx - 1]
                    y = ys[idx]
                    new_sk = sorted(list(yearly_skill[y] - yearly_skill[prev_y]))
                    gone_sk = sorted(list(yearly_skill[prev_y] - yearly_skill[y]))
                    rows.append({
                        "year": int(y),
                        "new_skills_top10": ", ".join(new_sk[:10]),
                        "gone_skills_top10": ", ".join(gone_sk[:10]),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("분석할 스킬 데이터가 없습니다.")
        except Exception:
            st.warning("신규/소멸 스킬 계산 중 오류가 발생했습니다.")

    # 직무 카테고리별 요구 역량 비교표
    st.markdown("**직무 카테고리별 요구 역량 비교표**")
    try:
        if not req_exp.empty:
            pivot = req_exp.groupby(["job_category", "skill"]).size().reset_index(name="count")
            top_by_job = pivot.sort_values(["job_category", "count"], ascending=[True, False]).groupby("job_category").head(7)
            st.dataframe(top_by_job, use_container_width=True)
        else:
            st.info("분석할 스킬 데이터가 없습니다.")
    except Exception:
        st.warning("직무별 요구 역량 비교표 계산 중 오류가 발생했습니다.")

    return req_exp


def render_preference_trends(df: pd.DataFrame):
    st.subheader("우대사항 트렌드")

    c1, c2 = st.columns(2)

    # 자격증 / 학력 / 경력 연수 요구 변화
    with c1:
        st.markdown("**자격증 / 학력 / 경력 연수 요구 변화**")
        try:
            exp_year = df.dropna(subset=["year"]).groupby("year")["min_experience_years"].mean().reset_index()
            cert_year = df.assign(cert_flag=df["preferred_certifications"].astype(str).str.strip().ne("")) \
                         .groupby("year")["cert_flag"].mean().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=exp_year["year"], y=exp_year["min_experience_years"], mode="lines+markers", name="평균 경력연수"))
            fig.add_trace(go.Bar(x=cert_year["year"], y=cert_year["cert_flag"], name="자격증 우대 비율", yaxis="y2", opacity=0.5))
            fig.update_layout(
                title="연도별 경력/자격증 요구 변화",
                yaxis=dict(title="평균 경력연수"),
                yaxis2=dict(title="자격증 우대 비율", overlaying="y", side="right", tickformat=".0%"),
                height=360,
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig, use_container_width=True)

            edu_dist = df.groupby(["year", "preferred_education"]).size().reset_index(name="count")
            st.plotly_chart(px.bar(edu_dist, x="year", y="count", color="preferred_education", barmode="stack", title="연도별 학력 요건 분포"), use_container_width=True)
        except Exception:
            st.warning("우대사항 변화 계산 중 오류가 발생했습니다.")

    # 산업군별 우대사항 차이 + AI/데이터 급증 여부
    with c2:
        st.markdown("**산업군별 우대사항 차이 / AI·데이터 급증 여부**")
        try:
            pref_exp = explode_skills(df, "preferred_skills")
            if not pref_exp.empty:
                top_pref = pref_exp.groupby(["industry", "skill"]).size().reset_index(name="count")
                top_pref = top_pref.sort_values(["industry", "count"], ascending=[True, False]).groupby("industry").head(5)
                st.dataframe(top_pref, use_container_width=True)

            ai_pattern = r"(AI|데이터|Data|ML|머신러닝|LLM|RAG)"
            ai_flag = (
                df["preferred_text"].astype(str).str.contains(ai_pattern, case=False, regex=True)
                | df["preferred_skills"].astype(str).str.contains(ai_pattern, case=False, regex=True)
                | df["required_skills"].astype(str).str.contains(ai_pattern, case=False, regex=True)
            )
            ai_year = df.assign(ai_data_pref=ai_flag).groupby("year")["ai_data_pref"].mean().reset_index()
            fig_ai = px.line(ai_year, x="year", y="ai_data_pref", markers=True, title="AI·데이터 관련 우대사항 비중")
            fig_ai.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_ai, use_container_width=True)

            if len(ai_year) >= 2:
                growth = ai_year.iloc[-1]["ai_data_pref"] - ai_year.iloc[0]["ai_data_pref"]
                if growth >= 0.15:
                    st.success("AI·데이터 관련 우대사항이 유의미하게 증가했습니다.")
                elif growth > 0:
                    st.info("AI·데이터 관련 우대사항이 완만히 증가하는 추세입니다.")
                else:
                    st.info("AI·데이터 관련 우대사항 비중이 정체 혹은 감소했습니다.")
        except Exception:
            st.warning("산업군 우대사항 분석 중 오류가 발생했습니다.")


def render_company_industry_insights(df: pd.DataFrame):
    st.subheader("기업·산업군별 인사이트")

    c1, c2 = st.columns(2)

    # 대기업 그룹사별 채용 성향 비교
    with c1:
        st.markdown("**대기업 그룹사별 채용 성향 비교**")
        try:
            grp = df.groupby(["company_group", "job_category"]).size().reset_index(name="count")
            fig = px.bar(grp, x="company_group", y="count", color="job_category", barmode="stack")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("그룹사별 채용 성향 계산 중 오류가 발생했습니다.")

    # 산업군별 요구 스킬 히트맵
    with c2:
        st.markdown("**산업군별 요구 스킬 히트맵**")
        try:
            req_exp = explode_skills(df, "required_skills")
            if req_exp.empty:
                st.info("히트맵을 그릴 스킬 데이터가 없습니다.")
            else:
                top_skills = req_exp["skill"].value_counts().head(12).index.tolist()
                h = req_exp[req_exp["skill"].isin(top_skills)].groupby(["industry", "skill"]).size().reset_index(name="count")
                pivot = h.pivot(index="industry", columns="skill", values="count").fillna(0)
                fig_h = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Teal")
                fig_h.update_layout(height=420)
                st.plotly_chart(fig_h, use_container_width=True)
        except Exception:
            st.warning("산업군 스킬 히트맵 생성 중 오류가 발생했습니다.")


def render_jobseeker_guide(df: pd.DataFrame, req_exp: pd.DataFrame):
    st.subheader("구직자 실전 가이드")

    jobs = ["전체"] + sorted(df["job_category"].dropna().astype(str).unique().tolist())
    selected_job = st.selectbox("직무 선택", jobs, index=0)

    top5 = suggest_top5_skills(req_exp, selected_job)
    st.markdown("**직무별 지금 준비해야 할 Top 5 스킬 (체크리스트)**")
    if top5.empty:
        st.info("Top 5 스킬을 계산하기 위한 데이터가 부족합니다.")
    else:
        for _, row in top5.iterrows():
            st.checkbox(f"{row['skill']} (우선순위 점수: {int(row['score'])})", value=False)

    st.markdown("**2026 채용 전망**")
    forecast_msg, yearly = forecast_2026_postings(df)
    st.info(forecast_msg)

    if not yearly.empty:
        st.plotly_chart(px.line(yearly, x="year", y="postings", markers=True, title="연도별 공고 수 기반 전망 참고"), use_container_width=True)


def render_raw_summary_table(df: pd.DataFrame):
    st.subheader("원데이터 요약 테이블")

    try:
        summary = {
            "총 공고 수": len(df),
            "기간(최소 연도)": int(df["year"].dropna().min()) if df["year"].dropna().size else "미상",
            "기간(최대 연도)": int(df["year"].dropna().max()) if df["year"].dropna().size else "미상",
            "기업 수": int(df["company"].nunique()),
            "산업군 수": int(df["industry"].nunique()),
            "직무 카테고리 수": int(df["job_category"].nunique()),
            "평균 경력연수": round(float(df["min_experience_years"].mean()), 2),
        }
        st.dataframe(pd.DataFrame([summary]), use_container_width=True)
        st.dataframe(df.head(50), use_container_width=True)
    except Exception:
        st.warning("원데이터 요약 테이블 생성 중 오류가 발생했습니다.")


# -------------------------------
# [4] Optional Google Sheet staging
# -------------------------------
def upload_to_google_sheet_if_requested(df: pd.DataFrame):
    """
    Approval-based upload placeholder.
    Sensitive credentials must come from st.secrets or environment variables.
    """
    st.markdown("---")
    st.markdown("### 승인 기반 구글 시트 적재 (선택)")

    with st.expander("구글 시트 적재 설정", expanded=False):
        approved = st.checkbox("분석 결과를 구글 시트에 적재하겠습니다 (승인)", value=False)
        sheet_key = st.text_input("Google Sheet Key", value=os.getenv("GOOGLE_SHEET_KEY", ""))
        worksheet_name = st.text_input("Worksheet 이름", value="on_top_results")

        if approved and st.button("구글 시트로 적재 실행"):
            try:
                import gspread
                from google.oauth2.service_account import Credentials

                svc_info = st.secrets.get("gcp_service_account", None)
                if svc_info is None:
                    st.error("`st.secrets['gcp_service_account']`가 없어 적재를 진행할 수 없습니다.")
                    return

                scopes = [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive",
                ]
                creds = Credentials.from_service_account_info(svc_info, scopes=scopes)
                gc = gspread.authorize(creds)

                if not sheet_key:
                    st.error("Google Sheet Key를 입력해주세요.")
                    return

                sh = gc.open_by_key(sheet_key)
                try:
                    ws = sh.worksheet(worksheet_name)
                    ws.clear()
                except Exception:
                    ws = sh.add_worksheet(title=worksheet_name, rows="2000", cols="30")

                safe_df = df.copy()
                safe_df["posting_date"] = safe_df["posting_date"].astype(str)
                ws.update([safe_df.columns.tolist()] + safe_df.astype(str).values.tolist())
                st.success("구글 시트 적재 완료")
            except Exception as e:
                st.warning(f"구글 시트 적재 중 오류가 발생했습니다: {e}")


def main():
    st.title("온톱(On-Top) 채용 트렌드 분석 대시보드")
    st.caption("고용센터 상담사용: 채용공고 기반 Skill-Shift 분석 및 실전 컨설팅 인사이트")

    st.markdown(
        """
        **사용 방법**
        1. `data` 폴더 PDF 자동분석 또는 CSV 업로드/데모 선택
        2. 필터로 연도/산업군/직무를 선택
        3. 트렌드와 역량 인사이트를 확인
        4. 필요 시 승인 후 구글 시트에 결과 적재
        """
    )

    # Input area (PDF-first for counselors)
    pdf_files = find_pdf_files("data")
    with st.container(border=True):
        st.markdown("### 데이터 입력")
        source = st.radio(
            "데이터 소스 선택",
            options=["data 폴더 PDF 자동분석", "CSV 업로드", "데모 데이터"],
            index=0 if pdf_files else 2,
            horizontal=True,
        )
        uploaded = st.file_uploader("채용공고 CSV 업로드", type=["csv"], disabled=(source != "CSV 업로드"))
        st.caption(f"`data` 폴더에서 발견한 PDF 수: {len(pdf_files)}")
        if source == "data 폴더 PDF 자동분석" and not PDF_READER_AVAILABLE:
            st.warning("PyPDF2를 불러오지 못해 파일명 기반 메타데이터만 분석합니다.")

    ingest_report = pd.DataFrame()
    try:
        if source == "data 폴더 PDF 자동분석":
            raw, ingest_report = load_pdf_dataset("data")
            if raw.empty:
                st.warning("`data` 폴더 PDF를 읽지 못해 데모 데이터를 사용합니다.")
                raw = generate_demo_data()
        elif source == "CSV 업로드" and uploaded is not None:
            raw = pd.read_csv(uploaded)
        elif source == "CSV 업로드" and uploaded is None:
            st.info("CSV 파일을 업로드해주세요.")
            return
        else:
            raw = generate_demo_data()
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return

    # Standardization pipeline
    try:
        df = normalize_column_names(raw)
        df = safe_parse_date(df)
        df = ensure_required_columns(df)

        # Fill missing half values if date exists
        missing_half = df["half"].isna() | (df["half"].astype(str).str.strip() == "")
        df.loc[missing_half & df["posting_date"].notna(), "half"] = np.where(
            df.loc[missing_half & df["posting_date"].notna(), "posting_date"].dt.month <= 6,
            "상반기",
            "하반기",
        )
        df["half"] = df["half"].fillna("미상")
    except Exception as e:
        st.error(f"전처리 중 오류가 발생했습니다: {e}")
        return

    selected_years, selected_industries, selected_jobs = sidebar_filters(df)
    fdf = apply_filters(df, selected_years, selected_industries, selected_jobs)

    if fdf.empty:
        st.warning("선택한 필터 조건에 맞는 데이터가 없습니다. 필터를 조정해주세요.")
        return

    if source == "data 폴더 PDF 자동분석":
        with st.expander("PDF 파싱 결과 확인", expanded=False):
            c1, c2, c3 = st.columns(3)
            ok_count = int((ingest_report["status"] == "ok").sum()) if not ingest_report.empty else 0
            c1.metric("파싱 성공 파일", ok_count)
            c2.metric("파싱 실패 파일", max(len(ingest_report) - ok_count, 0))
            c3.metric("사용 데이터 행 수", len(df))
            if not ingest_report.empty:
                st.dataframe(ingest_report, use_container_width=True)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "요구 역량",
        "우대사항",
        "기업/산업 인사이트",
        "실전 가이드",
    ])

    with tab1:
        render_overview(fdf)

    with tab2:
        req_exp = render_skills_analysis(fdf)

    with tab3:
        render_preference_trends(fdf)

    with tab4:
        render_company_industry_insights(fdf)

    with tab5:
        req_exp = explode_skills(fdf, "required_skills")
        render_jobseeker_guide(fdf, req_exp)

    render_raw_summary_table(fdf)

    # Optional export stage (approval-based)
    upload_to_google_sheet_if_requested(fdf)


if __name__ == "__main__":
    main()
