from pathlib import Path

from dotenv import load_dotenv

from deepagents.backends import StateBackend
from langchain_classic.chains.llm_summarization_checker.base import PROMPTS_DIR
from langchain_core.messages import HumanMessage

from generation.agents.agent import create_agent
from generation.agents.validation_agent import create_company_claim_validator
from generation.tools.search import search_content
from src.indexing import *
import os

def main():
    load_dotenv()

    k = int(os.getenv('RETRIEVAL_K'))
    if k is None:
        k = 5

    config = EmbeddingConfig(
        'openai',
        openai_api_key= os.getenv('OPENAI_API_KEY'),
        openai_base_url= os.getenv('OPENAI_BASE_URL'),
        openai_model= os.getenv('OPENAI_MODEL'),

        bedrock_region_name= os.getenv('BEDROCK_REGION'),
        bedrock_profile_name= os.getenv('BEDROCK_PROFILE'),
        bedrock_model_id=os.getenv('BEDROCK_MODEL_ID')
    )
    embedding = get_embeddings(config)

    docs = load_context_docs("data/")
    docs = split_docs(docs)
    print(docs)

    retriever = create_retriever(docs, embedding, k=k)

    company_name = "Gunvor"
    role_title = "Graduate Program – Quantitative Analysis"
    job_description = r"""
Job Description:

Turn data into commercial insight.

At Gunvor, quantitative analysis plays a critical role in helping our commercial teams understand markets, identify opportunities and make informed trading decisions.

Our 18-month Quantitative Analysis Graduate Program is designed for curious, analytical graduates who want to apply mathematics, technology and data to real-world trading challenges while building a long-term career in commodity markets.

You'll join an international environment where quantitative research, analytics and commercial thinking come together to solve complex business problems.

Your Journey

During the Program, you'll complete two 9-month rotations, giving you exposure to different quantitative and commercial functions across the business.

Depending on business needs, rotations may include:

Quantitative Analysis

Market Risk

Research

Trading Analytics

You'll be based in Geneva, with the opportunity for an international rotation in one of our global offices, including Singapore, Houston or London.

Alongside your rotations, you'll follow a structured development journey combining technical learning, industry knowledge and professional development to prepare you for a career in quantitative analysis within commodity trading.

What You'll Do

Throughout the Program, you'll:

Build and enhance quantitative models that support commercial and trading decisions.

Analyse market data to identify trends, relationships and opportunities.

Develop forecasting, optimisation and analytical tools.

Work closely with quantitative analysts, traders, researchers and Market Risk teams.

Apply programming and statistical techniques to solve real business challenges.

Present analytical findings and recommendations to stakeholders.

Take ownership of meaningful projects from the beginning of your career.

Who We're Looking For

We're looking for analytical thinkers who enjoy solving complex problems and applying quantitative methods to commercial challenges.

You'll ideally have:

A Master's or PhD in Mathematics, Statistics, Physics, Engineering, Computer Science, Data Science, Quantitative Finance or another highly quantitative discipline.

Up to 24 months of professional experience, excluding internships.

Strong programming skills, particularly in Python.

Experience using modern analytical tools and AI-enabled solutions to enhance research, modelling or decision-making.

Excellent analytical, critical thinking and problem-solving skills.

Strong communication skills and the ability to explain complex ideas clearly.

Curiosity about global commodity markets and quantitative trading.

Fluency in English.

Previous internships or professional experience within quantitative finance, banking, commodities, energy trading or research will be considered a strong advantage.
    """,

    retrieved_context = retriever.invoke(
        f"{job_description}\n\nFind the most relevant candidate experiences based on this job description, "
        "skills, projects, and achievements"
    )

    candidate_evidence = "\n".join([
        f"- {doc.page_content}..."
        for doc in (retrieved_context if isinstance(retrieved_context, list) else [retrieved_context])
    ])

    with open(f"output/RAG_context.txt", "w") as f:
        f.write(candidate_evidence)

    # The below tools cannot be created because the current architecture loads context immediately into context window, instead of allowing the agent to delegate and control chunking retrieval. This is a limitation of the current implementation and will be addressed in future iterations.
    # retriever_tool = retrieve_content
    # bound_search = search_content.bind(retriever=retriever)
    PROMPTS: dict[str, str] = {}

    for path in Path('src/generation/prompts/').iterdir():
        if not path.is_dir():
            var_name = path.name
            content = path.read_text(encoding="utf-8")
            PROMPTS[var_name] = content

    SYSTEM_PROMPT = (
        # + "\n\n"
        # + "=" * 80
        # + "\n\n" +
        PROMPTS["sys_cover_letter.txt"].format()
    )

    internet_search = {"type": "web_search"}
    agent = create_agent(SYSTEM_PROMPT, [internet_search])


    HUMAN_QUERY = PROMPTS["human_prompt.txt"].format(
        job_description=job_description,
        company_name=company_name,
        role_title=role_title,
        retrieved_context=candidate_evidence,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=HUMAN_QUERY)]}
    )

    cover_letter = next(
        (msg.text for msg in reversed(result.get("messages", [])) if msg.text),
        None,
    )
    if cover_letter is None:
        raise RuntimeError("The cover-letter agent returned no text.")

    print(cover_letter)
    with open("output/cover_letter.txt", "w") as f:
        f.write(cover_letter)

    validator = create_company_claim_validator()
    validation_prompt = PROMPTS["company_claim_validation.txt"].format(
        company_name=company_name,
        role_title=role_title,
        cover_letter=cover_letter,
    )
    validation_result = validator.invoke(
        {"messages": [HumanMessage(content=validation_prompt)]}
    )
    validation_report = next(
        (
            msg.text
            for msg in reversed(validation_result.get("messages", []))
            if msg.text
        ),
        None,
    )
    if validation_report is None:
        raise RuntimeError("The company-claim validator returned no report.")

    with open("output/company_claim_validation.txt", "w") as f:
        f.write(validation_report)



if __name__ == "__main__":
    main()