import os
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

CHROMA_DIR = "chroma_db"
DATA_PATH = "/home/sergey/Desktop/LLM_Agro/data/Agriculture_Data_YSU_18.04.25.xlsx"


CROP_MAP_EN_TO_HY = {
    "grape": "խաղող", "grapes": "խաղող", "vineyard": "խաղող", "vineyards": "խաղող", "wine": "խաղող",
    "potato": "կարտոֆիլ", "potatoes": "կարտոֆիլ",
    "wheat": "ցորեն", "barley": "գարի", "corn": "եգիպտացորեն", "maize": "եգիպտացորեն",
    "oat": "վարսակ", "oats": "վարսակ",
    "pea": "ոլոռ", "peas": "ոլոռ", "bean": "լոբի", "beans": "լոբի",
    "lentil": "ոսպ", "lentils": "ոսպ", "chickpea": "սիսեռ", "chickpeas": "սիսեռ",
    "tobacco": "ծխախոտ", "flax": "կտավատ", "sunflower": "արևածաղիկ",
    "cabbage": "կաղամբ", "cucumber": "վարունգ", "tomato": "լոլիկ", "tomatoes": "լոլիկ",
    "eggplant": "սմբուկ", "pepper": "տաքդեղ", "beetroot": "ճակնդեղ", "beet": "ճակնդեղ",
    "carrot": "գազար", "onion": "սոխ", "garlic": "սխտոր",
    "green pea": "կանաչ ոլոռ", "green bean": "կանաչ լոբի",
    "vegetable": "բանջարեղեն", "vegetables": "բանջարեղեն",
    "melon": "բոստանային", "melon crops": "բոստանային",
    "fodder roots": "կերի արմատ",
    "pome fruits": "հնդավոր մրգեր", "stone fruits": "կորիզավոր մրգեր",
    "nuts": "ընկուզապտուղ", "subtropical fruits": "մերձարևադարձային", "berries": "հատապտուղ",
}

REGION_MAP_HY_TO_EN = {
    "Արագածոտն": "Aragatsotn",
    "Արարատ": "Ararat",
    "Արմավիր": "Armavir",
    "Գեղարքունիք": "Gegharkunik",
    "Լոռի": "Lori",
    "Կոտայք": "Kotayk",
    "Շիրակ": "Shirak",
    "Սյունիք": "Syunik",
    "Վայոց ձոր": "Vayots Dzor",
    "Տավուշ": "Tavush",
    "Հայաստանի Հանրապետություն, ընդամենը": "Republic of Armenia, total",
}

REGION_MAP_EN_TO_HY = {v.lower(): k for k, v in REGION_MAP_HY_TO_EN.items()}

REGION_COORDS = {
    "Aragatsotn": (40.45, 44.25),
    "Ararat": (39.95, 44.55),
    "Armavir": (40.15, 44.05),
    "Gegharkunik": (40.35, 45.30),
    "Lori": (40.95, 44.50),
    "Kotayk": (40.35, 44.70),
    "Shirak": (40.80, 43.85),
    "Syunik": (39.25, 46.30),
    "Vayots Dzor": (39.75, 45.45),
    "Tavush": (40.95, 45.15),
}

CLIMATE_COLUMN_MAP = {
    "Ջրտուք": "Irrigation indicator",
    "օդի միջին հարաբերական խոնավություն(%)": "Average relative humidity (%)",
    "մթնոլորտային ճնշում(հՊա)": "Atmospheric pressure (hPa)",
    "տեղումների քանակ(մմ)": "Precipitation (mm)",
    "օդի միջին ջերմաստիճան(°C)": "Average air temperature (°C)",
}


RAG_PROMPT = """
You are AgroAdvisor AI, an agricultural assistant.

Answer using ONLY the provided PDF context.

Rules:
- Always answer in English only.
- Be clear and practical.
- Do not invent facts.
- If the context is insufficient, say that the knowledge base does not contain enough information.
- Add a short caution when giving agricultural advice.

Context:
{context}

Question:
{question}

Answer:
"""


SMART_PROMPT = """
You are AgroAdvisor AI, an advanced agricultural decision-support assistant for Armenia.

You must combine:
1. Retrieved PDF knowledge base context
2. Statistical evidence from the Armenian agricultural dataset
3. Suitability scores, crop climate profiles, risk level, scenario results, and explainability notes when available

Rules:
- Always answer in English only.
- Do not use Armenian words in the final answer.
- Do not invent facts.
- Treat irrigation values as historical dataset indicators, not guaranteed current water availability.
- Do not assume irrigation units unless explicitly stated.
- Use Final Suitability Score as the primary ranking when provided.
- Explain recommendations using yield, climate match, irrigation indicators, risk level, and PDF agronomic reasoning.
- If the user gives constraints such as limited water, hot year, dry year, cold year, or target crop, use them in the recommendation.
- End with a short caution that this is decision-support information and local field assessment is recommended.
- Use the detected intent to structure the answer.
- If the intent is region_to_crop, recommend crops for the detected region.
- If the intent is crop_to_region, recommend the best regions for the detected crop.
- If the intent is suitability_check, directly say whether the crop-region combination is suitable and explain why.
- If the intent is climate_scenario, explain how the scenario changes suitability or risk.
- If the intent is climate_risk, focus on risk reduction and safer crop choices.

PDF context:
{context}

Dataset evidence:
{data_summary}

User question:
{question}

Answer:
"""
REGION_TO_CROP_PROMPT = """
You are AgroAdvisor AI.

The user wants crop recommendations for a specific Armenian region.

Use:
1. Dataset evidence for best-performing crops, climate indicators, and risk level
2. PDF context for agronomic reasoning, climate-smart agriculture, irrigation, soil, and adaptation

Answer in English only.
Do not invent facts.
Treat irrigation values as historical indicators, not guaranteed water availability.

PDF context:
{context}

Dataset evidence:
{data_summary}

Question:
{question}

Answer:
"""


CROP_TO_REGION_PROMPT = """
You are AgroAdvisor AI.

The user wants to know the best Armenian region for a specific crop.

Use the Final Suitability Score as the main ranking.
Explain the result using yield, climate match, irrigation indicator, risk level, and PDF context.

Answer in English only.
Do not invent facts.

PDF context:
{context}

Dataset evidence:
{data_summary}

Question:
{question}

Answer:
"""


SUITABILITY_CHECK_PROMPT = """
You are AgroAdvisor AI.

The user is asking whether a specific crop is suitable for a specific region.

Give a direct answer first: Suitable / Moderately suitable / Not ideal.
Then explain using dataset evidence, suitability score, climate indicators, risk level, and PDF context.

Answer in English only.
Do not invent facts.

PDF context:
{context}

Dataset evidence:
{data_summary}

Question:
{question}

Answer:
"""


CLIMATE_RISK_PROMPT = """
You are AgroAdvisor AI.

The user is asking about agricultural risk under climate or water stress.

Focus on:
- climate risk
- water risk
- safer crop choices
- adaptation strategies from the PDF knowledge base
- dataset evidence from the selected region or crop
- Do not describe any crop as heat-tolerant, drought-resistant, or water-stress tolerant unless this is explicitly supported by the PDF context.
- If a crop appears in the dataset only because of high historical yield, say "historically high-performing", not "climate-resilient".
- For dry or hot conditions, separate "historical performance" from "climate risk suitability".

Answer in English only.
Do not invent facts.
Do not say a crop is drought-resistant unless the evidence supports it.

PDF context:
{context}

Dataset evidence:
{data_summary}

Question:
{question}

Answer:
"""


GENERAL_ADVISOR_PROMPT = SMART_PROMPT

st.set_page_config(page_title="AgroAdvisor AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1220 45%, #060b12 100%);
}
.block-container { padding-top: 2rem; max-width: 1450px; }
h1, h2, h3 { color: #f8fafc; }
[data-testid="stMetricValue"] { color: #ffffff; }
.stButton > button {
    background: linear-gradient(90deg, #0077ff, #00a3ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-weight: 700;
}
.stTextInput input, .stTextArea textarea, .stSelectbox div {
    background-color: #111827;
    color: #ffffff;
    border-radius: 8px;
}
.advisor-card {
    border: 1px solid #1f6feb;
    border-radius: 12px;
    padding: 18px;
    background: rgba(15, 23, 42, 0.75);
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    df["Մարզ"] = df["Մարզ"].astype(str).str.strip()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def crop_to_hy(crop_english):
    return CROP_MAP_EN_TO_HY.get(crop_english.lower().strip(), crop_english)


def detect_crop_from_question(question):
    q = question.lower()
    for crop_en in sorted(CROP_MAP_EN_TO_HY.keys(), key=len, reverse=True):
        if crop_en in q:
            return crop_en.title(), CROP_MAP_EN_TO_HY[crop_en]
    return None, None


def detect_region_from_question(question):
    q = question.lower()
    for region_en_lower, region_hy in REGION_MAP_EN_TO_HY.items():
        if region_en_lower in q:
            return REGION_MAP_HY_TO_EN[region_hy]
    return None


def detect_climate_condition(question):
    q = question.lower()
    conditions = []

    if any(w in q for w in ["dry", "drought", "low rainfall", "water shortage", "limited water", "water limited"]):
        conditions.append("dry_or_water_limited")

    if any(w in q for w in ["hot", "heat", "high temperature", "warmer"]):
        conditions.append("hot")

    if any(w in q for w in ["cold", "frost", "low temperature", "cool"]):
        conditions.append("cold_or_frost")

    if any(w in q for w in ["humid", "humidity", "wet"]):
        conditions.append("humid_or_wet")

    if any(w in q for w in ["rainy", "high rainfall", "precipitation", "heavy rain"]):
        conditions.append("rainfall_related")

    return conditions

def detect_intent(question):
    q = question.lower()

    if any(w in q for w in ["compare", "versus", "vs", "difference between"]):
        return "region_comparison"

    if any(w in q for w in ["if rainfall", "if temperature", "scenario", "decreases", "increases", "+2", "-20%"]):
        return "climate_scenario"

    if any(w in q for w in ["is", "suitable", "can i grow", "can i cultivate"]):
        crop_en, _ = detect_crop_from_question(question)
        region_en = detect_region_from_question(question)

        if crop_en and region_en:
            return "suitability_check"

    if any(w in q for w in ["where", "which region", "best region"]):
        crop_en, _ = detect_crop_from_question(question)

        if crop_en:
            return "crop_to_region"

    if any(w in q for w in ["what crops", "what can i grow", "what can i cultivate", "i live in"]):
        region_en = detect_region_from_question(question)

        if region_en:
            return "region_to_crop"

    if any(w in q for w in ["risk", "drought", "hot", "dry", "limited water", "water shortage", "frost"]):
        return "climate_risk"

    return "general_knowledge"

def search_columns(df, keyword):
    keyword = keyword.lower().strip()
    return [col for col in df.columns if keyword in col.lower()]


def get_yield_column(df, crop_keyword):
    cols = search_columns(df, crop_keyword)
    yield_cols = [c for c in cols if "1 հեկտարի միջին բերքատվությունը" in c]
    if not yield_cols:
        return None
    return yield_cols[0]


def get_climate_average_by_region(df):
    climate_cols = list(CLIMATE_COLUMN_MAP.keys())
    data = df[df["Մարզ"] != "Հայաստանի Հանրապետություն, ընդամենը"].copy()

    for col in climate_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    result = data.groupby("Մարզ")[climate_cols].mean().round(2).reset_index()
    result["Մարզ"] = result["Մարզ"].replace(REGION_MAP_HY_TO_EN)
    result = result.rename(columns={"Մարզ": "Region"})
    result = result.rename(columns=CLIMATE_COLUMN_MAP)

    return result


def get_top_regions_by_yield(df, crop_keyword, top_n=5):
    col = get_yield_column(df, crop_keyword)
    if col is None:
        return None, None

    data = df[df["Մարզ"] != "Հայաստանի Հանրապետություն, ընդամենը"].copy()
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[col])

    result = data.groupby("Մարզ")[col].mean().sort_values(ascending=False).round(2).reset_index()
    result.columns = ["Region", "Average Yield (quintals per hectare)"]
    result["Region"] = result["Region"].replace(REGION_MAP_HY_TO_EN)

    return col, result.head(top_n)


def get_top_crops_for_region(df, region_en, top_n=5):
    region_hy = REGION_MAP_EN_TO_HY.get(region_en.lower(), region_en)
    region_data = df[df["Մարզ"] == region_hy].copy()

    if region_data.empty:
        return None

    rows = []
    used_hy = set()

    for crop_en, crop_hy in CROP_MAP_EN_TO_HY.items():
        if crop_hy in used_hy:
            continue

        col = get_yield_column(df, crop_hy)
        if col is None:
            continue

        region_data[col] = pd.to_numeric(region_data[col], errors="coerce")
        avg_yield = region_data[col].mean()

        if pd.notna(avg_yield):
            rows.append({
                "Crop": crop_en.title(),
                "Average Yield (quintals per hectare)": round(avg_yield, 2)
            })
            used_hy.add(crop_hy)

    if not rows:
        return None

    result = pd.DataFrame(rows).sort_values("Average Yield (quintals per hectare)", ascending=False).head(top_n)
    result = result.reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result


def build_crop_climate_profile(df, crop_keyword):
    col = get_yield_column(df, crop_keyword)
    if col is None:
        return None

    climate_raw_cols = list(CLIMATE_COLUMN_MAP.keys())
    data = df[df["Մարզ"] != "Հայաստանի Հանրապետություն, ընդամենը"].copy()

    data[col] = pd.to_numeric(data[col], errors="coerce")
    for c in climate_raw_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=[col])
    if data.empty:
        return None

    threshold = data[col].quantile(0.75)
    top_data = data[data[col] >= threshold]

    profile = top_data[climate_raw_cols].mean().round(2).to_dict()
    profile = {CLIMATE_COLUMN_MAP[k]: v for k, v in profile.items()}

    return profile


def calculate_crop_region_suitability(df, crop_keyword, top_n=10, scenario=None):
    col = get_yield_column(df, crop_keyword)
    if col is None:
        return None

    _, yield_df = get_top_regions_by_yield(df, crop_keyword, top_n=10)
    if yield_df is None:
        return None

    climate_df = get_climate_average_by_region(df)
    result = yield_df.merge(climate_df, on="Region", how="left")

    profile = build_crop_climate_profile(df, crop_keyword)

    if profile:
        target_temp = profile.get("Average air temperature (°C)")
        target_precip = profile.get("Precipitation (mm)")
        target_humidity = profile.get("Average relative humidity (%)")
        target_irrigation = profile.get("Irrigation indicator")
    else:
        target_temp = result["Average air temperature (°C)"].mean()
        target_precip = result["Precipitation (mm)"].mean()
        target_humidity = result["Average relative humidity (%)"].mean()
        target_irrigation = result["Irrigation indicator"].mean()

    if scenario:
        if "temp_change" in scenario:
            result["Scenario temperature (°C)"] = result["Average air temperature (°C)"] + scenario["temp_change"]
        else:
            result["Scenario temperature (°C)"] = result["Average air temperature (°C)"]

        if "rainfall_change_pct" in scenario:
            result["Scenario precipitation (mm)"] = result["Precipitation (mm)"] * (1 + scenario["rainfall_change_pct"] / 100)
        else:
            result["Scenario precipitation (mm)"] = result["Precipitation (mm)"]
    else:
        result["Scenario temperature (°C)"] = result["Average air temperature (°C)"]
        result["Scenario precipitation (mm)"] = result["Precipitation (mm)"]

    max_yield = result["Average Yield (quintals per hectare)"].max()
    result["Yield Score"] = result["Average Yield (quintals per hectare)"] / max_yield * 100

    def safe_match(value, target):
        if target is None or target == 0 or pd.isna(value) or pd.isna(target):
            return 50
        return max(0, 100 - abs(value - target) / abs(target) * 100)

    result["Temperature Match"] = result["Scenario temperature (°C)"].apply(lambda x: safe_match(x, target_temp))
    result["Rainfall Match"] = result["Scenario precipitation (mm)"].apply(lambda x: safe_match(x, target_precip))
    result["Humidity Match"] = result["Average relative humidity (%)"].apply(lambda x: safe_match(x, target_humidity))
    result["Irrigation Match"] = result["Irrigation indicator"].apply(lambda x: safe_match(x, target_irrigation))

    result["Climate Score"] = (
        result["Temperature Match"] * 0.40
        + result["Rainfall Match"] * 0.35
        + result["Humidity Match"] * 0.25
    )

    result["Water Score"] = result["Irrigation Match"]

    result["Final Suitability Score"] = (
        result["Yield Score"] * 0.50
        + result["Climate Score"] * 0.35
        + result["Water Score"] * 0.15
    )

    result["Risk Level"] = result["Final Suitability Score"].apply(
        lambda x: "Low" if x >= 75 else ("Medium" if x >= 50 else "High")
    )

    result["Explainability"] = result.apply(
        lambda r: (
            f"Yield contributes {round(r['Yield Score'], 1)}/100; "
            f"climate match contributes {round(r['Climate Score'], 1)}/100; "
            f"irrigation indicator match contributes {round(r['Water Score'], 1)}/100."
        ),
        axis=1
    )

    cols = [
        "Region",
        "Average Yield (quintals per hectare)",
        "Average air temperature (°C)",
        "Precipitation (mm)",
        "Average relative humidity (%)",
        "Irrigation indicator",
        "Yield Score",
        "Climate Score",
        "Water Score",
        "Final Suitability Score",
        "Risk Level",
        "Explainability",
    ]

    if scenario:
        cols.insert(3, "Scenario temperature (°C)")
        cols.insert(5, "Scenario precipitation (mm)")

    result = result[cols].round(2)
    result = result.sort_values("Final Suitability Score", ascending=False).head(top_n).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))

    return result


def extract_scenario(question):
    q = question.lower()
    scenario = {}

    if "+2" in q or "2°c" in q or "2 c" in q or "2 degrees" in q:
        scenario["temp_change"] = 2

    if "-20%" in q or "20% less" in q or "rainfall decreases by 20" in q:
        scenario["rainfall_change_pct"] = -20

    if "dry" in q or "drought" in q or "low rainfall" in q:
        scenario["rainfall_change_pct"] = scenario.get("rainfall_change_pct", -20)

    if "hot" in q or "heat" in q:
        scenario["temp_change"] = scenario.get("temp_change", 2)

    return scenario if scenario else None


def get_trend_data(df, region_en, crop_english):
    region_hy = REGION_MAP_EN_TO_HY.get(region_en.lower(), region_en)
    crop_keyword = crop_to_hy(crop_english)
    col = get_yield_column(df, crop_keyword)

    if col is None:
        return None, None

    trend = df[df["Մարզ"] == region_hy][["Տարի", col]].copy()
    trend[col] = pd.to_numeric(trend[col], errors="coerce")
    trend = trend.dropna().sort_values("Տարի").reset_index(drop=True)
    trend.columns = ["Year", "Yield (quintals per hectare)"]

    return col, trend


def compare_regions_for_crop(df, crop_keyword, regions):
    suitability = calculate_crop_region_suitability(df, crop_keyword, top_n=10)
    if suitability is None:
        return None
    return suitability[suitability["Region"].isin(regions)]


def build_dataset_evidence(question):
    df = load_data()

    intent = detect_intent(question)
    detected_crop_en, crop_keyword = detect_crop_from_question(question)
    detected_region_en = detect_region_from_question(question)
    climate_conditions = detect_climate_condition(question)
    scenario = extract_scenario(question)

    evidence_parts = []

    evidence_parts.append(f"Detected intent: {intent}")

    if detected_crop_en:
        profile = build_crop_climate_profile(df, crop_keyword)
        suitability = calculate_crop_region_suitability(df, crop_keyword, top_n=5, scenario=scenario)

        evidence_parts.append(f"Detected crop: {detected_crop_en}")

        if profile:
            evidence_parts.append(f"""
Crop climate profile estimated from high-performing historical observations:
{pd.DataFrame([profile]).to_string(index=False)}
""")

        if scenario:
            evidence_parts.append(f"""
Detected climate scenario:
{scenario}
""")

        if suitability is not None:
            evidence_parts.append(f"""
Crop-region suitability ranking:
{suitability.to_string(index=False)}
""".replace("suıtability", "suitability"))

    if detected_region_en:
        top_crops = get_top_crops_for_region(df, detected_region_en, top_n=5)
        climate_df = get_climate_average_by_region(df)
        region_climate = climate_df[climate_df["Region"] == detected_region_en]

        evidence_parts.append(f"Detected region: {detected_region_en}")

        if top_crops is not None:
            evidence_parts.append(f"""
Best-performing crops in this region based on historical average yield:
{top_crops.to_string(index=False)}
""")

        if not region_climate.empty:
            evidence_parts.append(f"""
Average climate indicators for {detected_region_en}:
{region_climate.to_string(index=False)}
""")

    if climate_conditions:
        evidence_parts.append(f"""
Detected climate or water condition:
{", ".join(climate_conditions)}
Use this condition as a risk factor.
""")

    if not evidence_parts:
        evidence_parts.append("""
No specific crop, Armenian region, or climate scenario was detected.
Use the PDF knowledge base for a general agricultural answer.
""")

    return "\n".join(evidence_parts), detected_crop_en, detected_region_en, intent


def get_pdf_context(question, k=5):
    db = load_vector_db()
    results = db.similarity_search_with_score(question, k=k)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    sources = []

    for doc, score in results:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        if isinstance(page, int):
            page += 1

        sources.append({
            "source": os.path.basename(source),
            "page": page,
            "score": round(score, 4),
            "preview": doc.page_content[:300].replace("\n", " ")
        })

    return context_text, sources


def ask_llm(prompt_template, question, context, data_summary=""):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    final_prompt = prompt.format(
        question=question,
        context=context,
        data_summary=data_summary
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.invoke(final_prompt).content


def generate_rag_answer(question):
    context, sources = get_pdf_context(question, k=4)
    answer = ask_llm(RAG_PROMPT, question, context)
    return answer, sources

def choose_prompt_by_intent(intent):
    if intent == "region_to_crop":
        return REGION_TO_CROP_PROMPT

    if intent == "crop_to_region":
        return CROP_TO_REGION_PROMPT

    if intent == "suitability_check":
        return SUITABILITY_CHECK_PROMPT

    if intent in ["climate_risk", "climate_scenario"]:
        return CLIMATE_RISK_PROMPT

    return GENERAL_ADVISOR_PROMPT

def generate_smart_advice(question):
    data_summary, detected_crop_en, detected_region_en, intent = build_dataset_evidence(question)

    context, sources = get_pdf_context(question, k=5)

    selected_prompt = choose_prompt_by_intent(intent)

    answer = ask_llm(
        selected_prompt,
        question,
        context,
        data_summary
    )

    return answer, data_summary, sources, detected_crop_en, detected_region_en, intent


st.title("🌱 AgroAdvisor AI")
st.caption("AI-Powered Agricultural Decision Support System for Armenia")

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Knowledge Base",
    "📊 Data Explorer",
    "🤖 Smart Advisor",
    "🌡️ Advanced Tools",
])


# -----------------------------
# Tab 1: Knowledge Base
# -----------------------------

with tab1:
    st.subheader("Knowledge Base Chat")
    st.write("Ask agriculture-related questions based on the PDF knowledge base.")

    q = st.text_input(
        "Ask a PDF-based agriculture question:",
        "How does climate change affect water management in agriculture?"
    )

    if st.button("Generate Answer"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching PDF knowledge base..."):
                answer, sources = generate_rag_answer(q)

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            for i, src in enumerate(sources, 1):
                with st.expander(f"{i}. {src['source']} | Page {src['page']}"):
                    st.write(f"Similarity score: {src['score']}")
                    st.write(src["preview"])


# -----------------------------
# Tab 2: Data Explorer
# -----------------------------

with tab2:
    st.subheader("Data Explorer")

    section = st.radio(
        "Choose section:",
        ["Dataset Analytics", "Trend Analysis"],
        horizontal=True
    )

    df = load_data()

    if section == "Dataset Analytics":
        st.write("Dataset overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Years", f"{df['Տարի'].min()}–{df['Տարի'].max()}")

        st.divider()

        analysis_type = st.selectbox(
            "Choose analysis:",
            ["Top regions by crop yield", "Climate averages by region"]
        )

        if analysis_type == "Top regions by crop yield":
            crop = st.text_input("Crop in English:", "Grape")

            if st.button("Run Yield Analysis"):
                _, res = get_top_regions_by_yield(
                    df,
                    crop_to_hy(crop),
                    top_n=5
                )

                if res is None:
                    st.error("No yield column found.")
                else:
                    res = res.copy()
                    res.insert(0, "Rank", range(1, len(res) + 1))
                    st.dataframe(
                        res,
                        use_container_width=True,
                        hide_index=True
                    )

        elif analysis_type == "Climate averages by region":
            if st.button("Show Climate Averages"):
                climate_result = get_climate_average_by_region(df)
                st.dataframe(
                    climate_result,
                    use_container_width=True,
                    hide_index=True
                )

    elif section == "Trend Analysis":
        st.write("Analyze crop yield changes by region and year.")

        regions_en = sorted([
            REGION_MAP_HY_TO_EN[r]
            for r in df["Մարզ"].unique()
            if r != "Հայաստանի Հանրապետություն, ընդամենը"
        ])

        region = st.selectbox(
            "Select region:",
            regions_en
        )

        crop = st.text_input(
            "Crop in English:",
            "Grape",
            key="trend_crop"
        )

        _, trend = get_trend_data(df, region, crop)

        if trend is None:
            st.warning("No yield data found.")
        else:
            fig = px.line(
                trend,
                x="Year",
                y="Yield (quintals per hectare)",
                markers=True,
                title=f"{crop} yield trend in {region}"
            )

            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)

            avg_value = trend["Yield (quintals per hectare)"].mean()
            min_value = trend["Yield (quintals per hectare)"].min()
            max_value = trend["Yield (quintals per hectare)"].max()

            first_value = trend["Yield (quintals per hectare)"].iloc[0]
            last_value = trend["Yield (quintals per hectare)"].iloc[-1]

            change = ((last_value - first_value) / first_value) * 100

            c1.metric("Average", round(avg_value, 2))
            c2.metric("Minimum", round(min_value, 2))
            c3.metric("Maximum", round(max_value, 2))
            c4.metric("Change %", f"{round(change, 2)}%")

            st.dataframe(
                trend,
                use_container_width=True,
                hide_index=True
            )


# -----------------------------
# Tab 3: Smart Advisor
# -----------------------------

with tab3:
    st.subheader("Smart Advisor")

    st.info(
        "Ask a free-form agricultural question. The system combines PDF knowledge, dataset evidence, suitability scores, climate profiles, and risk analysis."
    )

    question = st.text_area(
        "Ask any decision-support question:",
        "I live in Shirak. What crops can I cultivate?"
    )

    if st.button("✨ Generate Smart Advice"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Combining PDFs, dataset, scores, risk, and scenario logic..."):
                advice, data_summary, sources, _, _, intent = generate_smart_advice(
                question
            )

            st.markdown(
                f"""
                <div class="advisor-card">
                    <h3>🎯 Smart Advice</h3>
                    <p>{advice}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f"**Detected intent:** `{intent}`")
            st.markdown("### 📊 Dataset Evidence")
            st.code(data_summary)

            st.markdown("### 📖 PDF Sources")
            for i, src in enumerate(sources, 1):
                with st.expander(f"{i}. {src['source']} | Page {src['page']}"):
                    st.write(f"Similarity score: {src['score']}")
                    st.write(src["preview"])


# -----------------------------
# Tab 4: Advanced Tools
# -----------------------------

with tab4:
    st.subheader("Advanced Tools")

    tool = st.radio(
        "Choose tool:",
        ["Region Comparison", "Climate Scenario Simulator", "Map View"],
        horizontal=True
    )

    df = load_data()

    if tool == "Region Comparison":
        st.write("Compare selected regions for a specific crop.")

        crop = st.text_input(
            "Crop in English:",
            "Grape",
            key="compare_crop"
        )

        regions = sorted([
            REGION_MAP_HY_TO_EN[r]
            for r in df["Մարզ"].unique()
            if r != "Հայաստանի Հանրապետություն, ընդամենը"
        ])

        selected_regions = st.multiselect(
            "Choose regions:",
            regions,
            default=["Ararat", "Armavir"]
        )

        if st.button("Compare Regions"):
            comparison = compare_regions_for_crop(
                df,
                crop_to_hy(crop),
                selected_regions
            )

            if comparison is None or comparison.empty:
                st.warning("No comparison data found.")
            else:
                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

                fig = px.bar(
                    comparison,
                    x="Region",
                    y="Final Suitability Score",
                    color="Risk Level",
                    title=f"{crop} suitability comparison"
                )

                st.plotly_chart(fig, use_container_width=True)

    elif tool == "Climate Scenario Simulator":
        st.write("Simulate how temperature and rainfall changes affect crop-region suitability.")

        crop = st.text_input(
            "Crop in English:",
            "Grape",
            key="scenario_crop"
        )

        temp_change = st.slider(
            "Temperature change (°C):",
            min_value=-5,
            max_value=5,
            value=2
        )

        rainfall_change = st.slider(
            "Rainfall change (%):",
            min_value=-50,
            max_value=50,
            value=-20
        )

        if st.button("Run Scenario"):
            scenario = {
                "temp_change": temp_change,
                "rainfall_change_pct": rainfall_change
            }

            table = calculate_crop_region_suitability(
                df,
                crop_to_hy(crop),
                top_n=10,
                scenario=scenario
            )

            if table is None:
                st.warning("No scenario data found.")
            else:
                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True
                )

                fig = px.bar(
                    table,
                    x="Region",
                    y="Final Suitability Score",
                    color="Risk Level",
                    title=f"{crop} suitability under climate scenario"
                )

                st.plotly_chart(fig, use_container_width=True)

    elif tool == "Map View":
        st.write("View crop suitability by Armenian region.")

        crop = st.text_input(
            "Crop in English:",
            "Grape",
            key="map_crop"
        )

        table = calculate_crop_region_suitability(
            df,
            crop_to_hy(crop),
            top_n=10
        )

        if table is None:
            st.warning("No map data found.")
        else:
            map_df = table.copy()

            map_df["lat"] = map_df["Region"].map(
                lambda r: REGION_COORDS.get(r, (None, None))[0]
            )

            map_df["lon"] = map_df["Region"].map(
                lambda r: REGION_COORDS.get(r, (None, None))[1]
            )

            map_df = map_df.dropna(subset=["lat", "lon"])

            fig = px.scatter_geo(
                map_df,
                lat="lat",
                lon="lon",
                size="Final Suitability Score",
                color="Risk Level",
                hover_name="Region",
                hover_data=[
                    "Average Yield (quintals per hectare)",
                    "Final Suitability Score",
                    "Risk Level"
                ],
                title=f"{crop} suitability map by region"
            )

            fig.update_geos(
                projection_type="natural earth",
                showcountries=True,
                countrycolor="gray",
                showland=True,
                landcolor="rgb(20, 30, 45)",
                showocean=True,
                oceancolor="rgb(5, 10, 20)",
                lataxis_range=[38.5, 41.5],
                lonaxis_range=[43.3, 47.0],
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                map_df.drop(columns=["lat", "lon"]),
                use_container_width=True,
                hide_index=True
            )