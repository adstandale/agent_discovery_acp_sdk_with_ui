import os
from typing import TypedDict, Annotated
import operator
import asyncio
import uuid
from datetime import datetime, timezone

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import(
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,   
)

from deep_eval import evaluate_agent_response

from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
from fastapi import FastAPI
from langfuse.langchain import CallbackHandler

from acp_sdk.client import Client


def get_trace_id() -> str:
    trace_id = uuid.uuid4().hex
    os.environ["LANGFUSE_TRACE_ID"] = trace_id
    return trace_id


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)

DATABASE_URL=os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add it to your .env file before running the app.")


CREW_SERVER_URL = os.getenv("CREW_SERVER_URL", "http://127.0.0.1:8002")


def safe_print_text(value: str) -> str:
    return value.encode("ascii", "replace").decode("ascii")


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    country_name: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    
def flight_agent(state: TravelState):
    query=state["user_query"]
    flight_data=search_flights(query)
    
    return {
         "flight_results": flight_data,
         "messages":[
             AIMessage(content=f"Flight search results for query '{query}':\n{flight_data}")
             
         ],
         "llm_calls": state.get("llm_calls", 0) + 1       
}
    
def hotel_agent(state: TravelState):
    query=f"Best hotels for {state['user_query']}"
    hotel_results=tavily_search(query)
    
    return{
        "hotel_results": hotel_results,
        "messages":[
            AIMessage(content=f"Hotel search results for query '{query}':\n{hotel_results}")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1    
        
    }
    
def itinerary_agent(state: TravelState):
    prompt=f"""
    create a travel itinerary.
    User query: {state['user_query']}
    
    Flight results: {state['flight_results']}
    
    Hotel results: {state['hotel_results']}
    """
    
    response=llm.invoke([
        SystemMessage(
            content="You are a travel assistant that creates travel itineraries based on user queries and search results."
        ),
        HumanMessage(content=prompt)
        
    ])
    evaluate_agent_response(
        agent_name="itinerary_agent",
        user_input=state.get("user_query", ""),
        response=response.content,
        expected_output="A clear, structured travel itinerary that matches the user's trip and the flight/hotel search context.",
        context=[
            f"User query: {state.get('user_query', '')}",
            f"Flight results: {state.get('flight_results', '')}",
            f"Hotel results: {state.get('hotel_results', '')}",
        ],
    )
    return{
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1 
    }
    
def final_agent(state: TravelState):
    final_prompt=f"""
    Generate a final travel itinerary based on the following information:
    
    Flights: {state['flight_results']}
    Hotels: {state['hotel_results']}
    Itinerary: {state['itinerary']}
    """
    response=llm.invoke([
        HumanMessage(content=final_prompt)
    ])
    evaluate_agent_response(
        agent_name="final_agent",
        user_input=state.get("user_query", ""),
        response=response.content,
        expected_output="A final travel response that integrates flights, hotel information, and the itinerary into a usable recommendation.",
        context=[
            f"Flights: {state.get('flight_results', '')}",
            f"Hotels: {state.get('hotel_results', '')}",
            f"Itinerary: {state.get('itinerary', '')}",
        ],
    )
    
    return{
        "messages":[response],        
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def legal_guide_agent(state: TravelState):
    country_name = state.get("country_name", "").strip()
    if not country_name:
        raise ValueError("country_name is required")
    legal_guide = call_acp_crew(country_name)

    return {
        "messages": [AIMessage(content=f"Legal guide for '{country_name}':\n{legal_guide}")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
    
def call_acp_crew(country_name: str) -> str:
    request_time = datetime.now(timezone.utc).isoformat()

    ACP_TIMELINE.append({
        "direction": "main_to_crew",
        "sender": "main_langgraph_agent",
        "receiver": "crew_legal_agent",
        "protocol": "ACP SDK",
        "endpoint": f"{CREW_SERVER_URL.rstrip('/')}/runs",
        "method": "run_sync",
        "country": country_name,
        "status": "request_sent",
        "timestamp": request_time,
    })

    async def call():
        async with Client(
            base_url=CREW_SERVER_URL,
            headers={"Content-Type": "application/json"},
        ) as client:
            run = await client.run_sync(
                country_name,
                agent="crew_legal_agent",
            )
            run.raise_for_status()

            if not run.output:
                return "Crew ACP agent returned no legal guide."

            text_parts = []
            for message in run.output:
                for part in message.parts:
                    if part.content:
                        text_parts.append(part.content)

            result_text = "\n".join(text_parts)
            completion_status = "completed" if "ACP status: completed" in result_text else str(run.status)

            if completion_status == "completed":
                ACP_TIMELINE.append({
                    "direction": "crew_to_main",
                    "sender": "crew_legal_agent",
                    "receiver": "main_langgraph_agent",
                    "protocol": "ACP SDK",
                    "endpoint": f"{CREW_SERVER_URL.rstrip('/')}/runs",
                    "method": "completion_notice",
                    "run_id": str(run.run_id),
                    "country": country_name,
                    "status": "completed",
                    "message": "Legal guide generation completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            return result_text

    return asyncio.run(call())
    
graph = StateGraph(TravelState)    

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)
graph.add_node("legal_guide_agent", legal_guide_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", "legal_guide_agent")
graph.add_edge("legal_guide_agent", END)

_conn=psycopg.connect(DATABASE_URL, autocommit=True)
checkpointer=PostgresSaver(_conn)
checkpointer.setup()

app=graph.compile(checkpointer=checkpointer)

from agent_discovery import get_main_agent_cards

ACP_TIMELINE: list[dict] = []

api_app = FastAPI(title="Travel Main Agent", version="1.0")


@api_app.get("/")
def root():
    return {
        "agent": "main_agent",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "run_workflow": "/run",
        "crew_protocol": "ACP SDK",
        "crew_agent": "crew_legal_agent",
    }


@api_app.get("/agents/discovery")
def main_agent_discovery():
    main_file = os.path.join(os.path.dirname(__file__), "main.py")
    return {
        "server": "main_langgraph_agent",
        "url": "http://127.0.0.1:8001",
        "agents": get_main_agent_cards(main_file),
    }


@api_app.get("/acp/timeline")
def acp_timeline(limit: int = 100):
    return {"messages": list(reversed(ACP_TIMELINE[-limit:]))}


@api_app.get("/health")
def health():
    return {"status": "ok", "agent": "main_agent"}


@api_app.post("/run")
def run_travel(request: dict):
    country_name = request.get("country_name", "").strip()
    user_input = request.get("user_query", "").strip()

    if not country_name:
        raise ValueError("country_name is required")
    if not user_input:
        raise ValueError("user_query is required")

    trace_id = get_trace_id()

    try:
        langfuse_handler = CallbackHandler(trace_context={"trace_id": trace_id})
    except Exception:
        langfuse_handler = None

    config = {
        "configurable": {
            "thread_id": request.get("thread_id", "user_anandprakash"),
            "langfuse_trace_id": trace_id,
        }
    }
    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "country_name": country_name,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    final_messages = [safe_print_text(msg.content) for msg in result["messages"]]
    return {
        "trace_id": trace_id,
        "messages": final_messages,
        "final_response": final_messages[-1] if final_messages else "",
    }


if __name__=="__main__":
    country_name = input("Enter the country name: ").strip()
    if not country_name:
        raise ValueError("Country name is required")

    user_input = input("Enter your travel query: ").strip()
    if not user_input:
        raise ValueError("Travel query is required")

    trace_id = get_trace_id()
    langfuse_handler = None

    try:
        langfuse_handler = CallbackHandler(trace_context={"trace_id": trace_id})
    except Exception:
        langfuse_handler = None

    config={
        "configurable": {
            "thread_id": "user_anandprakash",
            "langfuse_trace_id": trace_id,
        }
    }

    if langfuse_handler is not None:
        config["callbacks"] = [langfuse_handler]

    result=app.invoke(
        {
        "messages":[
            HumanMessage(content=user_input)
        ],
        "user_query": user_input,
        "country_name": country_name,
        "flight_results": "",
        "hotel_results": "",
        "itinerary": "",
        "llm_calls": 0
        },
        config=config
    )
    
    print("\n\nFINAL RESPONSE\n\n")
    
    final_output_path = os.path.join(os.path.dirname(__file__), "final_reponse.txt")

    with open(final_output_path, "w", encoding="utf-8") as final_file:
        final_file.write("FINAL RESPONSE\n\n")

        for msg in result["messages"]:
            text = safe_print_text(msg.content)
            print(text)
            final_file.write(text + "\n")

    print(f"\nFinal response saved to: {final_output_path}")

    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass
