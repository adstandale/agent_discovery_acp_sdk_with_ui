import os
import requests
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage
from main import app

st.set_page_config(page_title="Multi-Agent Travel Assistant", layout="wide")
st.title("Multi-Agent Travel Assistant")

with st.sidebar:
    st.header("Server Configuration")
    main_url = st.text_input("Main agent URL", "http://127.0.0.1:8001")
    crew_url = st.text_input("Crew agent URL", "http://127.0.0.1:8002")

    if st.button("Refresh server status"):
        st.rerun()


def get_health(url: str, agent_name: str, acp: bool = False):
    try:
        if acp:
            ping_response = requests.get(f"{url.rstrip('/')}/ping", timeout=3)
            ping_response.raise_for_status()

            agents_response = requests.get(f"{url.rstrip('/')}/agents", timeout=3)
            agents_response.raise_for_status()
            agent_names = [agent.get("name") for agent in agents_response.json().get("agents", [])]

            if "crew_legal_agent" not in agent_names:
                raise RuntimeError("crew_legal_agent is not registered")
        else:
            response = requests.get(f"{url.rstrip('/')}/health", timeout=3)
            response.raise_for_status()

        st.success(f"{agent_name}: online")
        return True
    except Exception as exc:
        st.error(f"{agent_name}: offline ({exc.__class__.__name__})")
        return False


status_col1, status_col2 = st.columns(2)
with status_col1:
    main_online = get_health(main_url, "Main Agent")
with status_col2:
    crew_online = get_health(crew_url, "Crew Agent", acp=True)

st.divider()
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Start Travel Workflow")
    country_name = st.text_input("Country", placeholder="Enter country name")
    user_query = st.text_area("Travel request", placeholder="Enter your travel request")
    thread_id = st.text_input("Thread ID", "user_anandprakash")

    inputs_ready = bool(country_name.strip() and user_query.strip())
    if not inputs_ready:
        st.caption("Enter both a country and travel request to run the workflow.")

    if st.button("Run Multi-Agent Workflow", type="primary", disabled=not main_online or not inputs_ready):
        st.session_state["workflow_running"] = True
        with st.spinner("Running flight, hotel, itinerary, legal-guide, and final agents..."):
            try:
                response = requests.post(
                    f"{main_url.rstrip('/')}/run",
                    json={
                        "country_name": country_name,
                        "user_query": user_query,
                        "thread_id": thread_id,
                    },
                    timeout=300,
                )
                response.raise_for_status()
                st.session_state["workflow_result"] = response.json()
                st.session_state["workflow_running"] = False
                st.success("Workflow completed")
            except requests.exceptions.Timeout:
                st.session_state["workflow_running"] = False
                st.error("The workflow timed out. The crew agent may still be processing.")
            except Exception as exc:
                st.session_state["workflow_running"] = False
                st.error(f"Workflow failed: {exc.__class__.__name__}: {exc}")

with right_col:
    st.subheader("Final Response")
    result = st.session_state.get("workflow_result")
    if result:
        st.caption(f"Trace ID: {result.get('trace_id', 'unknown')}")
        st.markdown(result.get("final_response", "No final response received."))

st.divider()
st.subheader("ACP Bidirectional Communication")

if st.session_state.get("workflow_running"):
    st.info("Communication in progress — refreshing live ACP events...")

try:
    ping_response = requests.get(f"{crew_url.rstrip('/')}/ping", timeout=5)
    ping_response.raise_for_status()

    agents_response = requests.get(f"{crew_url.rstrip('/')}/agents", timeout=5)
    agents_response.raise_for_status()

    agents = agents_response.json().get("agents", [])
    st.success("ACP protocol connected")

    st.caption("Main/LangGraph → ACP SDK → Crew AI")
    st.caption("Crew AI → ACP SDK run response → Main/LangGraph")

    if agents:
        st.caption("Registered ACP agents")
        st.json(agents)

    timeline_response = requests.get(
        f"{main_url.rstrip('/')}/acp/timeline",
        params={"limit": 100},
        timeout=5,
    )
    timeline_response.raise_for_status()
    timeline = timeline_response.json().get("messages", [])

    if not timeline:
        st.info("No ACP communication has occurred yet.")
    else:
        for event in timeline:
            if event.get("direction") == "main_to_crew":
                title = "Main/LangGraph → Crew: run_sync request"
            else:
                title = "Crew → Main/LangGraph: run_sync response"

            with st.expander(title, expanded=False):
                st.caption(event.get("timestamp", ""))
                st.json(event)
except Exception as exc:
    st.info(f"ACP timeline unavailable: {exc.__class__.__name__}: {exc}")

if st.button("Refresh ACP communication"):
    st.rerun()

if st.session_state.get("workflow_result"):
    st.divider()
    st.subheader("Full Workflow Messages")
    for index, message in enumerate(st.session_state["workflow_result"].get("messages", []), start=1):
        st.markdown(f"**Message {index}**")
        st.text(message)

st.divider()
st.subheader("Discovered Agent Cards")
main_card, crew_card = st.columns(2)


def render_discovered_cards(container, discovery_url: str, server_name: str):
    try:
        response = requests.get(discovery_url, timeout=5)
        response.raise_for_status()
        discovery = response.json()

        container.success(f"{server_name}: online")
        container.caption(f"Server: {discovery.get('url', 'unknown')}")

        cards = discovery.get("agents", [])
        if not cards:
            container.info("No discoverable agent functions found.")
            return

        for card in cards:
            with container.container(border=True):
                container.markdown(f"### {card['name']}")
                container.caption(f"Agent ID: `{card['id']}`")
                container.write(card["description"])
                container.caption(f"Source: `{card['source']}:{card['line']}`")

                functions = card.get("functions", [])
                calls = card.get("calls", [])
                if functions:
                    container.markdown("**Functions**")
                    container.code("\n".join(functions), language="python")
                if calls:
                    container.markdown("**Calls / dependencies**")
                    container.write(", ".join(f"`{call}()`" for call in calls))
    except Exception as exc:
        container.error(f"{server_name}: discovery unavailable ({exc.__class__.__name__})")


with main_card:
    render_discovered_cards(main_card, f"{main_url.rstrip('/')}/agents/discovery", "Main LangGraph Server")

with crew_card:
    render_discovered_cards(crew_card, f"{crew_url.rstrip('/')}/discovery/agents", "Crew ACP Server")

