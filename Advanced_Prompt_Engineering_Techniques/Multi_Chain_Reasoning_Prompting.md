# **Multi-Chain Reasoning Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Multi-Chain Reasoning Prompting (MCR, also called Meta-Chain Reasoning) is an advanced way to ask a Large Language Model (LLM) to solve a problem. Instead of asking the LLM for a single answer, you first ask it to generate multiple different thinking processes (called "Chain-of-Thought" paths) for the same problem. Once it has these multiple chains, it performs a second-level, or "meta," reasoning step. 

This means the model acts as a smart editor or synthesizer: it reviews and compares all the different steps it generated to form a final improved answer. Thus it can correct cases where none of the single chains were perfect, but by combining pieces of different chains you can get a better result. 

![Multi Chain Reasoning prompting](images/3-multi-chain-prompt.jpg)

Figure from [Multi Chain Reasoning prompting](https://arxiv.org/abs/2304.13007) paper. 

## **Prompt Template**

Here is the generation prompt template for multi chain reasoning prompting.

```
You are a careful step-by-step reasoning assistant.

Question: {question}

Instructions:
- Think step by step.
- Produce a clear and logically consistent chain of thought.
- Then provide the final answer in free-form text.
```
Here is the meta reasoning prompt template for multi chain reasoning prompting.

```
You are a meta-reasoning assistant.

You are given multiple reasoning chains generated independently for the
same question. Your task is to:

1. Compare all reasoning chains.
2. Identify errors, inconsistencies, or weaknesses.
3. Combine the correct reasoning steps from all chains to form a
   refined, more robust meta-reasoning.
4. Produce the final free-form answer.

Question:
{question}

Reasoning Chains:
{all_chains}

Instructions:
- Perform explicit meta-analysis.
- Synthesize the best reasoning from all chains.
- Return your combined reasoning and final answer in this JSON format:

{{
  "meta_reasoning": "...",
  "answer": "..."
}}
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Zero-Shot Implementation**

Now let's see the implementation of zero-shot multi-chain reasoning promtping technique using LangChain v1.0

```python
# ---------------------------------------------------------
# Zero-Shot Multi-Chain Reasoning Prompting (MCR)
# ---------------------------------------------------------

# pip install langchain langchain-google-genai pydantic

import os
import time
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. Set your Gemini API key
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ---------------------------------------------------------
# 2. Structured model for each reasoning chain (free-form answer)
# ---------------------------------------------------------
class MCRCandidate(BaseModel):
    reasoning_chain: str = Field(
        ..., description="Full reasoning steps used to derive the answer"
    )
    answer: str = Field(
        ..., description="Final free-form answer (not restricted to numeric)"
    )


parser = PydanticOutputParser(pydantic_object=MCRCandidate)


# ---------------------------------------------------------
# 3. Initialize Gemini model with sampling enabled
# ---------------------------------------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.8,
    top_k=40,
)


# ---------------------------------------------------------
# 4. Zero-shot chain generation prompt
# ---------------------------------------------------------
generation_prompt_template = ChatPromptTemplate.from_template(
    """
You are a careful step-by-step reasoning assistant.

Question: {question}

Instructions:
- Think step by step.
- Produce a clear and logically consistent chain of thought.
- Then provide the final answer in free-form text.

Return output in the following JSON format:
{format_instructions}
"""
)

generation_prompt = generation_prompt_template.partial(
    format_instructions=parser.get_format_instructions()
)

gen_chain = generation_prompt | model | parser


# ---------------------------------------------------------
# 5. Meta-Reasoning Combination Prompt
# ---------------------------------------------------------
meta_prompt = ChatPromptTemplate.from_template(
    """
You are a meta-reasoning assistant.

You are given multiple reasoning chains generated independently for the
same question. Your task is to:

1. Compare all reasoning chains.
2. Identify errors, inconsistencies, or weaknesses.
3. Combine the correct reasoning steps from all chains to form a
   refined, more robust meta-reasoning.
4. Produce the final free-form answer.

Question:
{question}

Reasoning Chains:
{all_chains}

Instructions:
- Perform explicit meta-analysis.
- Synthesize the best reasoning from all chains.
- Return your combined reasoning and final answer in this JSON format:

{{
  "meta_reasoning": "...",
  "answer": "..."
}}
"""
)

meta_chain = meta_prompt | model


# ---------------------------------------------------------
# 6. Multi-Chain Reasoning main function
# ---------------------------------------------------------
def multi_chain_reasoning(question: str, n_samples: int = 3):
    candidates = []

    # ---- Stage 1: Generate multiple independent reasoning chains ----
    for _ in range(n_samples):
        out = gen_chain.invoke({"question": question})
        candidates.append(out)
        time.sleep(1)

    # Prepare formatted reasoning chains for meta-prompt
    chain_text = ""
    for i, c in enumerate(candidates, 1):
        chain_text += (
            f"\n[{i}] Reasoning:\n{c.reasoning_chain}\nFinal Answer: {c.answer}\n"
        )

    # ---- Stage 2: Meta-combine reasoning chains ----
    meta_output = meta_chain.invoke(
        {
            "question": question,
            "all_chains": chain_text,
        }
    )

    return meta_output, candidates


# ---------------------------------------------------------
# 7. Run MCR on the updated example question
# ---------------------------------------------------------
question = (
    "A train leaves at 8:15 AM and takes 4 hours and 35 minutes "
    "to reach its destination. If the destination city is 2 hours "
    "ahead of the starting city's time zone, what time is it at the "
    "destination city when the train arrives?"
)

meta_output, all_candidates = multi_chain_reasoning(question, n_samples=3)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------
print("\n===== META CHAIN REASONING OUTPUT =====")
print(meta_output.content)

print("\n===== ALL GENERATED CANDIDATE CHAINS =====")
for i, c in enumerate(all_candidates, 1):
    print(f"\n--- Candidate {i} ---")
    print(c.reasoning_chain)
    print("Answer:", c.answer)
```

Here the output is
```

===== META CHAIN REASONING OUTPUT =====
{
  "meta_reasoning": "All three reasoning chains correctly identify the necessary steps to solve the problem. They all begin by calculating the train's arrival time in the starting city's time zone. This is done by adding the travel duration (4 hours and 35 minutes) to the departure time (8:15 AM). \n\nStarting with 8:15 AM, adding 4 hours results in 12:15 PM. Then, adding 35 minutes to 12:15 PM yields an arrival time of 12:50 PM in the starting city's time zone.\n\nNext, all chains correctly account for the time zone difference. The problem states the destination city is 2 hours 'ahead' of the starting city's time zone. Therefore, 2 hours must be added to the arrival time calculated in the starting city's time zone.\n\nAdding 2 hours to 12:50 PM results in 2:50 PM.\n\nAll chains consistently follow these steps and arrive at the same correct final answer. There are no errors, inconsistencies, or weaknesses found across the chains; they are all sound and demonstrate a clear understanding of time calculations and time zone adjustments.",
  "answer": "The train arrives at 2:50 PM in the destination city."
}

===== ALL GENERATED CANDIDATE CHAINS =====

--- Candidate 1 ---
1. **Determine the departure time:** The train leaves at 8:15 AM.
2. **Calculate the travel duration:** The journey takes 4 hours and 35 minutes.
3. **Calculate the arrival time in the starting city's time zone:**
    *   Add 4 hours to 8:15 AM: 8:15 AM + 4 hours = 12:15 PM.
    *   Add 35 minutes to 12:15 PM: 12:15 PM + 35 minutes = 12:50 PM.
    *   So, the train arrives at 12:50 PM in the starting city's time zone.
4. **Adjust for the time zone difference:** The destination city is 2 hours ahead of the starting city.
    *   Add 2 hours to the arrival time calculated in step 3: 12:50 PM + 2 hours = 2:50 PM.
5. **State the final arrival time:** The train arrives at 2:50 PM in the destination city's time zone.
Answer: The train arrives at 2:50 PM in the destination city.

--- Candidate 2 ---
The train leaves at 8:15 AM. The travel duration is 4 hours and 35 minutes. First, we calculate the arrival time in the starting city's time zone. 
- Add 4 hours to 8:15 AM: 8:15 AM + 4 hours = 12:15 PM.
- Add 35 minutes to 12:15 PM: 12:15 PM + 35 minutes = 12:50 PM.
So, the train arrives at 12:50 PM in the starting city's time zone.
The destination city is 2 hours ahead of the starting city's time zone. Therefore, we need to add 2 hours to the calculated arrival time.
- 12:50 PM + 2 hours = 2:50 PM.
Thus, the train arrives at 2:50 PM in the destination city's time.
Answer: The train arrives at 2:50 PM at the destination city.

--- Candidate 3 ---
The train leaves at 8:15 AM. The travel duration is 4 hours and 35 minutes.

First, calculate the arrival time in the starting city's time zone:
Start Time: 8:15 AM
Add 4 hours: 8:15 AM + 4 hours = 12:15 PM
Add 35 minutes: 12:15 PM + 35 minutes = 12:50 PM
So, the train arrives at 12:50 PM in the starting city's time zone.

Next, adjust for the time zone difference. The destination city is 2 hours ahead of the starting city.
Arrival time in starting city's time zone: 12:50 PM
Add 2 hours for the time zone difference: 12:50 PM + 2 hours = 2:50 PM.

Therefore, the train arrives at 2:50 PM in the destination city's time.
Answer: The train arrives at 2:50 PM in the destination city.
```


## **Few-Shot Implementation**

Now let's see the implementation of few-shot multi-chain reasoning promtping technique using LangChain v1.0

```python
# ---------------------------------------------------------
# Few-Shot Multi-Chain Reasoning Prompting (MCR)
# ---------------------------------------------------------

# pip install langchain langchain-google-genai pydantic

import os
import time
from pydantic import BaseModel, Field
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ---------------------------------------------------------
# 1. Set Gemini API key
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ---------------------------------------------------------
# 2. Structured output schema for each chain
# ---------------------------------------------------------
class MCRCandidate(BaseModel):
    reasoning_chain: str = Field(
        ..., description="Full chain-of-thought reasoning"
    )
    answer: str = Field(
        ..., description="Final free-form answer"
    )


parser = PydanticOutputParser(pydantic_object=MCRCandidate)


# ---------------------------------------------------------
# 3. Initialize Gemini model with sampling enabled
# ---------------------------------------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.8,
    top_k=40,
)


# ---------------------------------------------------------
# 4. Few-shot example (train/time-zone problem)
# ---------------------------------------------------------
few_shot_example = """
Example Problem:
A train leaves at 8:15 AM and takes 4 hours and 35 minutes to reach its destination.
If the destination city is 2 hours ahead of the starting city's time zone,
what time is it at the destination city when the train arrives?

Example Chain-of-Thought:
First compute the arrival time in the starting city.
8:15 AM + 4 hours 35 minutes = 12:50 PM local time.
The destination city is 2 hours ahead, so convert 12:50 PM to destination time:
12:50 PM + 2 hours = 2:50 PM.
Thus, when the train arrives, the destination city's local time is 2:50 PM.

Example Final Answer:
2:50 PM at the destination city.
"""


# ---------------------------------------------------------
# 5. Few-shot generation prompt
# ---------------------------------------------------------
generation_prompt_template = ChatPromptTemplate.from_template(
    """
You are a detailed step-by-step reasoning assistant.

Below is a worked example:
{few_shot_example}

Now follow the same reasoning structure to answer the new question.

New Question:
{question}

Instructions:
- Provide a full chain-of-thought reasoning.
- Then give a concise final answer.
- Respond in this JSON format:
{format_instructions}
"""
)

generation_prompt = generation_prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

gen_chain = generation_prompt | model | parser


# ---------------------------------------------------------
# 6. Meta-Reasoning Combination Prompt
# ---------------------------------------------------------
meta_prompt = ChatPromptTemplate.from_template(
    """
You are a meta-reasoning assistant.

You are given multiple reasoning chains produced independently for the
same question. Your task is to:

1. Compare all reasoning chains.
2. Identify errors or inconsistencies.
3. Combine the correct pieces of reasoning into one improved, unified chain.
4. Produce the final free-form answer.

Question:
{question}

Candidate Reasoning Chains:
{all_chains}

Return your response in the following JSON format:

{{
  "meta_reasoning": "...",
  "answer": "..."
}}
"""
)

meta_chain = meta_prompt | model


# ---------------------------------------------------------
# 7. Few-Shot Multi-Chain Reasoning function (n_samples = 3)
# ---------------------------------------------------------
def multi_chain_reasoning(question: str, n_samples: int = 3):
    candidates = []

    # ---- Stage 1: Generate independent chains ----
    for _ in range(n_samples):
        out = gen_chain.invoke({"question": question})
        candidates.append(out)
        time.sleep(1)

    # Prepare reasoning block for meta prompt
    formatted = ""
    for idx, c in enumerate(candidates, 1):
        formatted += (
            f"\n[{idx}] Reasoning:\n{c.reasoning_chain}\nFinal Answer: {c.answer}\n"
        )

    # ---- Stage 2: Meta-synthesis ----
    meta_output = meta_chain.invoke(
        {"question": question, "all_chains": formatted}
    )

    return meta_output, candidates


# ---------------------------------------------------------
# 8. Run Few-shot Multi-Chain Reasoning
# ---------------------------------------------------------
question = (
    "Identify the capital city of the largest country in South America, "
    "and then state which continent that capital city is located on."
)

meta_output, all_outputs = multi_chain_reasoning(question, n_samples=3)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------
print("\n===== FINAL META-SYNTHESIZED ANSWER =====")
print(meta_output.content)

print("\n===== ALL GENERATED CANDIDATE CHAINS =====")
for i, out in enumerate(all_outputs, 1):
    print(f"\n--- Candidate {i} ---")
    print(out.reasoning_chain)
    print("Answer:", out.answer)
```

Here the output is

```

===== FINAL META-SYNTHESIZED ANSWER =====
{
  "meta_reasoning": "All three reasoning chains correctly identify Brazil as the largest country in South America, Brasília as its capital, and South America as the continent where Brasília is located. There are no factual errors or inconsistencies in the reasoning steps across the chains. The only minor difference lies in the phrasing of the final answer. Chains [2] and [3] provide a concise answer ('Brasília, South America'), while Chain [1] provides a more complete sentence that explicitly answers both parts of the question ('The capital city of the largest country in South America is Brasília, and it is located on the continent of South America.'). For a 'free-form answer' that fully addresses the prompt 'Identify... and then state...', Chain [1]'s final answer is slightly more complete and directly responsive to both clauses of the question. Therefore, the improved, unified chain will consolidate the consistent correct reasoning, and the final answer will adopt the more comprehensive phrasing from Chain [1].",
  "answer": "The capital city of the largest country in South America is Brasília, and it is located on the continent of South America."
}

===== ALL GENERATED CANDIDATE CHAINS =====

--- Candidate 1 ---
First, identify the largest country in South America. Brazil is the largest country in South America by both land area and population. Next, identify the capital city of Brazil, which is Brasília. Finally, state the continent where Brasília is located. Brasília is located within Brazil, which is on the continent of South America.
Answer: The capital city of the largest country in South America is Brasília, and it is located on the continent of South America.

--- Candidate 2 ---
First, identify the largest country in South America. The largest country in South America by both area and population is Brazil. Next, identify the capital city of Brazil. The capital city of Brazil is Brasília. Finally, state which continent Brasília is located on. Brasília is located on the continent of South America.
Answer: Brasília, South America.

--- Candidate 3 ---
First, identify the largest country in South America. Brazil is the largest country in South America by both land area and population. Next, identify the capital city of Brazil, which is Brasília. Finally, state the continent where Brasília is located. Brasília is located on the continent of South America.
Answer: Brasília, South America
```