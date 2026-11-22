"""
HR Screening Agent - Streamlit UI
LangGraph workflow: categorize experience -> assess skillset -> route to
HR interview / recruiter escalation / rejection.

Run locally:   streamlit run main.py
Deploy:        push main.py + requirements.txt to GitHub, deploy on Streamlit
               Community Cloud, and add OPENAI_API_KEY under App settings -> Secrets.
"""

import os
from typing import Literal

import streamlit as st
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HR Screening Agent", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 HR Screening Agent")
st.caption("Screens a job application for a **Python Developer** role using a LangGraph workflow.")


# ---------------------------------------------------------------------------
# API key: Streamlit secrets -> environment variable -> sidebar input
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass  # no secrets.toml file
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    return st.sidebar.text_input("OpenAI API key", type="password") or None


api_key = get_api_key()
if not api_key:
    st.info("Add your OpenAI API key in the sidebar (or in Streamlit secrets) to start.")
    st.stop()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str


# ---------------------------------------------------------------------------
# Build the graph once and cache it
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_app(key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=key)

    def categorize_experience(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            "Based on the following job application, categorize the candidate as "
            "'Entry-level', 'Mid-level' or 'Senior-level'. "
            "Respond with only the label, nothing else.\n"
            "Application : {application}"
        )
        chain = prompt | llm
        experience_level = chain.invoke({"application": state["application"]}).content.strip()
        return {"experience_level": experience_level}

    def assess_skillset(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            "Based on the job application for a Python Developer, assess the candidate's skillset. "
            "Respond with only 'Match' or 'No Match', nothing else.\n"
            "Application : {application}"
        )
        chain = prompt | llm
        skill_match = chain.invoke({"application": state["application"]}).content.strip()
        return {"skill_match": skill_match}

    def schedule_hr_interview(state: State) -> State:
        return {"response": "Candidate has been shortlisted for an HR interview."}

    def escalate_to_recruiter(state: State) -> State:
        return {"response": "Candidate has senior-level experience but doesn't match job skills."}

    def reject_application(state: State) -> State:
        return {"response": "Candidate doesn't meet JD and has been rejected."}

    def route_app(state: State) -> Literal[
        "schedule_hr_interview", "escalate_to_recruiter", "reject_application"
    ]:
        skill = state["skill_match"].strip().strip(".'\"").lower()
        level = state["experience_level"].strip().strip(".'\"").lower()
        if skill == "match":
            return "schedule_hr_interview"
        elif level == "senior-level":
            return "escalate_to_recruiter"
        else:
            return "reject_application"

    workflow = StateGraph(State)
    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")
    workflow.add_conditional_edges("assess_skillset", route_app)
    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


app = build_app(api_key)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
SAMPLES = {
    "— write your own —": "",
    "Senior Python dev (expect: HR interview)":
        "I have 10 years of experience in Python, Django, FastAPI and AWS. "
        "I have led backend teams and built data pipelines.",
    "Fresher, no Python (expect: rejection)":
        "I am a fresher with basic knowledge of HTML and CSS.",
    "Senior Java architect (expect: recruiter escalation)":
        "I have 12 years of experience as a Java architect working with Spring and microservices. "
        "I have not worked with Python.",
}

sample = st.selectbox("Try a sample application", list(SAMPLES.keys()))
application = st.text_area(
    "Candidate application",
    value=SAMPLES[sample],
    height=180,
    placeholder="Paste the candidate's application or resume summary here…",
)

if st.button("Screen candidate", type="primary", use_container_width=True):
    if not application.strip():
        st.warning("Please enter an application first.")
    else:
        with st.spinner("Screening the application…"):
            try:
                result = app.invoke({"application": application})
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        col1, col2 = st.columns(2)
        col1.metric("Experience level", result.get("experience_level", "—"))
        col2.metric("Skill match", result.get("skill_match", "—"))

        response = result.get("response", "")
        if "shortlisted" in response:
            st.success(f"✅ {response}")
        elif "senior-level" in response:
            st.warning(f"📨 {response}")
        else:
            st.error(f"❌ {response}")

        with st.expander("Raw workflow output"):
            st.json(result)


# ---------------------------------------------------------------------------
# Workflow diagram
# ---------------------------------------------------------------------------
with st.expander("View workflow graph"):
    try:
        st.image(app.get_graph().draw_mermaid_png())
    except Exception:
        # draw_mermaid_png calls an online renderer; fall back to Mermaid text
        st.code(app.get_graph().draw_mermaid(), language="text")
