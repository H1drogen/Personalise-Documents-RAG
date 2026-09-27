from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

def create_agent(instructions: str, tools):
    """Create a deep research agent with custom tools and prompts."""

    chunk_analyst_subagent = {
        "name": "chunk-analyst",
        "description": "Analyze documentation chunks from RAG retrieval.",
        "system_prompt": """Analyze retrieved documentation chunks and extract key facts.""",
    }

    background_researcher = {
        "name": "company-background-researcher",
        "description": "Research company background, mission, and market position.",
        "system_prompt": """Research company background and fundamentals.

    Use web_search to find:
    - Company mission, vision, and values
    - Industry and market position
    - Key products/services overview related to role
    """,
    }

    news_researcher = {
        "name": "company-news-researcher",
        "description": "Research recent company news and developments.",
        "system_prompt": """Research company recent news and developments.

    Use web_search to find:
    - Recent announcements and press releases
    - Product launches or updates

    Focus on last 6-12 months. Return under 300 words."""
    }

    model = init_chat_model(model="openai:gpt-5.5")

    agent = create_deep_agent(
        model=model,
        system_prompt=instructions,
        tools=tools,
        subagents=[
            background_researcher,
            news_researcher
        ],
    )

    return agent