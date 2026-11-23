import ast
from pathlib import Path
from typing import Any


def discover_agent_functions(source_file: str, owner: str) -> list[dict[str, Any]]:
    """Discover top-level agent functions and their direct function calls."""
    path = Path(source_file)
    if not path.exists():
        return []

    tree = ast.parse(path.read_text(encoding="utf-8"))
    agents: list[dict[str, Any]] = []

    role_map = {
        "flight_agent": ("Flight Search Agent", "Searches for flights matching the travel request."),
        "hotel_agent": ("Hotel Search Agent", "Searches for hotels matching the destination and trip."),
        "itinerary_agent": ("Itinerary Agent", "Creates a structured travel itinerary."),
        "final_agent": ("Final Response Agent", "Combines travel information into the final response."),
        "legal_guide_agent": ("Legal Guide Coordinator", "Calls the Crew ACP agent for legal guidance."),
        "call_acp_crew": ("ACP Communication Bridge", "Uses acp_sdk Client.run_sync() to call Crew."),
        "crew_legal_acp_agent": ("Crew Legal Agent", "Receives ACP runs and generates a legal handbook."),
        "search_legal_frameworks": ("Legal Research Function", "Searches official and credible legal sources."),
        "create_legal_guide": ("Legal Guide Generator", "Creates a structured country legal and compliance guide."),
    }

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        name = node.name
        if name not in role_map:
            continue

        calls = sorted({
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
        })

        title, description = role_map[name]
        agents.append({
            "id": name,
            "name": title,
            "owner": owner,
            "description": description,
            "functions": [name],
            "calls": calls,
            "source": str(path.name),
            "line": node.lineno,
            "status": "discovered",
        })

    return agents


def get_main_agent_cards(main_file: str) -> list[dict[str, Any]]:
    return discover_agent_functions(main_file, "main_langgraph_agent")


def get_crew_agent_cards(crew_file: str) -> list[dict[str, Any]]:
    return discover_agent_functions(crew_file, "crew_ai_agent")
