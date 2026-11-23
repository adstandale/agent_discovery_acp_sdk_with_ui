# legal_guide.py
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from acp_sdk import Message, MessagePart
from acp_sdk.server import agent, create_app
from agent_discovery import get_crew_agent_cards

from deep_eval import evaluate_agent_response

load_dotenv(Path(__file__).resolve().with_name(".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # Default to a specific model if not set
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY) if GROQ_API_KEY else None
search_tool = DuckDuckGoSearchRun(
    api_wrapper=DuckDuckGoSearchAPIWrapper(backend="lite", source="text", region="wt-wt")
)


def search_legal_frameworks(query: str) -> str:
    """Searches the web for legal documents, constitution details, and national laws for a given country."""
    try:
        return search_tool.run(query)
    except Exception as exc:
        return (
            f"Search tool failed for query '{query}': {exc}. "
            "Use official government and constitutional sources and general legal knowledge for the summary."
        )


def create_legal_guide(country_name: str, acp_message: dict | None = None) -> str:
    if llm is None:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the project's .env file before running the legal guide.")

    search_query = (
        f"{country_name} constitution legal system government structure laws legal framework official "
        "government sites"
    )
    search_results = search_legal_frameworks(search_query)

    research_prompt = f"""
    You are a senior legal researcher and analyst.
    Research the legal system of {country_name} using the provided search results.

    Requirements:
    - identify the form of government
    - find the country's constitution or foundational legal document
    - list at least five important legal domains
    - include official or credible sources when possible
    - summarize the government structure and key legal obligations

    Search results:
    {search_results}
    """

    research_response = llm.invoke([
        SystemMessage(content="You are a legal researcher producing a factual and structured legal summary."),
        HumanMessage(content=research_prompt),
    ])

    analysis_prompt = f"""
    Based on the research below, write a clean legal blueprint and compliance guide for {country_name}.
    Format it as markdown with sections:
    1. Executive Summary
    2. Constitutional Foundations
    3. Key Statutes and Legal Domains
    4. Business and Commercial Compliance
    5. Immigration, Residency, and Labor Rules
    6. Local Customs, Traffic, and Daily Compliance Notes
    7. Sources and References

    Research summary:
    {research_response.content}
    """

    final_response = llm.invoke([
        SystemMessage(content="You are a legal analyst and educator. Write a clear, professional, and practical handbook."),
        HumanMessage(content=analysis_prompt),
    ])

    evaluate_agent_response(
        agent_name="legal_guide_agent",
        user_input=f"Legal guide for {country_name}",
        response=final_response.content,
        expected_output=f"A structured legal guide for {country_name} covering the constitution, legal domains, business compliance, immigration, and local obligations.",
        context=[
            f"Country: {country_name}",
            f"Research summary: {research_response.content}",
        ],
    )

    return final_response.content

def target_country(target_country):
    return target_country


@agent(name="crew_legal_agent", description="Generates a country legal and compliance guide")
def crew_legal_acp_agent(input: list[Message]):
    country_name = ""

    if input:
        for message in input:
            for part in message.parts:
                if part.content:
                    country_name = part.content.strip()
                    break

    if not country_name:
        raise ValueError("country_name is required")

    legal_guide = create_legal_guide(country_name)
    return Message(
        role="agent",
        parts=[
            MessagePart(content="ACP status: completed", content_type="text/plain"),
            MessagePart(content=legal_guide, content_type="text/markdown"),
        ],
    )


acp_app = create_app(crew_legal_acp_agent)


@acp_app.get("/discovery/agents")
def crew_agent_discovery():
    crew_file = str(Path(__file__))
    return {
        "server": "crew_ai_agent",
        "url": "http://127.0.0.1:8002",
        "agents": get_crew_agent_cards(crew_file),
    }
