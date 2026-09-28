from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model


def create_company_claim_validator():
    """Create an agent that verifies factual claims about the target company."""
    validation_instructions = """You are a meticulous fact checker for company-specific claims in cover letters.

For every factual claim about the target company, its people, products, projects, events,
awards, or dates:
1. Search the web for corroborating evidence, prioritising the company's official website,
   official press releases, regulators, and reputable news sources.
2. Compare the claim with the evidence. Do not treat a search result snippet or an
   unverified secondary source as confirmation.
3. Report claims that are unsupported, contradicted, or have an incorrect date.

Pay particular attention to these categories:
- invented events (including events the candidate supposedly attended)
- invented companies, products, partnerships, or initiatives
- invented awards or misattributed awards
- incorrect dates for events, launches, announcements, achievements, or awards

Return a concise Markdown report with one row per issue in this table:
| Claim | Category | Status | Evidence | Source |

Use `UNSUPPORTED` when no reliable source confirms the claim, `CONTRADICTED` when
reliable evidence conflicts with it, and `SUPPORTED` only when the evidence clearly
matches. Include the exact source URL for every row. If no issues are found, say
"No unsupported company claims detected." Do not rewrite the letter.
"""


    model = init_chat_model(model="openai:gpt-5.5")

    agent = create_deep_agent(
        model=model,
        system_prompt=validation_instructions,
        tools=[{"type": "web_search"}],
        subagents=[],
    )

    return agent