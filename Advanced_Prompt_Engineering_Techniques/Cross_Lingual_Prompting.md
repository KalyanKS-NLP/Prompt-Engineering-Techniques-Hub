# **Cross Lingual Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Cross-Lingual Prompting (CLP) is a strategy designed to enhance the reasoning capabilities of Large Language Models (LLMs) when processing tasks in low-resource or non-dominant languages. While many LLMs possess vast knowledge bases, their ability to perform complex reasoning (such as arithmetic or logic) is often significantly stronger in high-resource languages like English compared to others.

CLP bridges this performance gap by separating the linguistic understanding from the logical reasoning. Instead of forcing the model to solve a complex problem directly in the source language where it may fail to reason correctly, the process is split into two distinct stages: Alignment (Translation) and Solving (Reasoning).

1. Cross-Lingual Alignment

In this step, the model is instructed to act as an expert in multilingual understanding. It receives the task in the source language (Ex: Hindi) and is asked to understand the given task step-by-step in the target language (English).

2. Task-Specific Solver

After clarifying the task in the target language, the model is instructed to switch roles and act as an expert in the target task (e.g., arithmetic reasoning, sentiment analysis, classification). It now solves the already-aligned problem entirely in the target language.


![Cross Lingual prompting](images/2-cross-lingual-prompt.jpg)

Figure from [Cross Lingual prompting](https://arxiv.org/abs/2310.14799) paper.

## **Prompt Template**

Here is the cross-lingual alignment prompt template for cross lingual prompting.

```
You are an expert in multi-lingual understanding in Hindi language.

Request: {task}

Let's understand the task in English step by step.

```

Here is the task-specific solver prompt template for cross lingual prompting.

```
You are an expert in arithmetic reasoning in English language.

Using the cross-lingual understanding provided below,
solve the task step by step in English.

Understanding:
{understanding}

Provide:
1. Step-by-step reasoning (in English)
2. The final numeric answer only
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of cross lingual promtping technique using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# ----------------------------------------------------------
# 1. Set Gemini API Key
# ----------------------------------------------------------

os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ----------------------------------------------------------
# 2. Structured Output Models
# ----------------------------------------------------------

class AlignedUnderstanding(BaseModel):
    understanding: str = Field(..., description="Step-by-step meaning of the Hindi task explained in English")


class TaskSolution(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning in English")
    final_answer: str = Field(..., description="Only the numeric final answer")


alignment_parser = PydanticOutputParser(pydantic_object=AlignedUnderstanding)
solution_parser = PydanticOutputParser(pydantic_object=TaskSolution)


# ----------------------------------------------------------
# 3. Initialize Gemini (gemini-2.5-flash)
# ----------------------------------------------------------

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)


# ----------------------------------------------------------
# 4. Prompt Templates
# ----------------------------------------------------------

# ------------------------------
# 4.1 Cross-Lingual Alignment Prompt
# ------------------------------
alignment_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert in multi-lingual understanding in Hindi language.

Request: {task}

Let's understand the task in English step by step.

Provide the output in this JSON format:
{format_instructions}
"""
)

alignment_prompt = alignment_prompt_template.partial(
    format_instructions=alignment_parser.get_format_instructions()
)


# ------------------------------
# 4.2 Task-Specific Solver Prompt
# ------------------------------
solver_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert in arithmetic reasoning in English language.

Using the cross-lingual understanding provided below,
solve the task step by step in English.

Understanding:
{understanding}

Provide:
1. Step-by-step reasoning (in English)
2. The final numeric answer only

Return your output in this JSON format:
{format_instructions}
"""
)

solver_prompt = solver_prompt_template.partial(
    format_instructions=solution_parser.get_format_instructions()
)


# ----------------------------------------------------------
# 5. Build LCEL Chains
# ----------------------------------------------------------

alignment_chain = alignment_prompt | model | alignment_parser
solver_chain = solver_prompt | model | solution_parser


# ----------------------------------------------------------
# 6. Run Cross-Lingual Prompting on Hindi Example
# ----------------------------------------------------------

task = "राम श्याम से तीन साल बड़ा है। अगर श्याम 10 साल का है, तो राम की उम्र कितनी है?"

# Step 1 — Cross-Lingual Alignment
alignment_result = alignment_chain.invoke({"task": task})
print("\n--- CROSS-LINGUAL UNDERSTANDING ---\n")
print(alignment_result.understanding)

# Step 2 — Task-Specific Solver
solution_result = solver_chain.invoke({
    "understanding": alignment_result.understanding
})
print("\n--- TASK REASONING ---\n")
print(solution_result.reasoning)

print("\n--- FINAL ANSWER ---\n")
print(solution_result.final_answer)
```
Here the output is
```
--- CROSS-LINGUAL UNDERSTANDING ---

The Hindi task can be broken down as follows:
1.  "राम श्याम से तीन साल बड़ा है।" translates to "Ram is three years older than Shyam."
2.  "अगर श्याम 10 साल का है," translates to "If Shyam is 10 years old,"
3.  "तो राम की उम्र कितनी है?" translates to "then what is Ram's age?"

In essence, the task asks to calculate Ram's age given that he is three years older than Shyam, and Shyam's current age is 10 years.

--- TASK REASONING ---

1. The problem states that Ram is three years older than Shyam. 2. It also provides Shyam's current age as 10 years. 3. To find Ram's age, we need to add the age difference (3 years) to Shyam's age (10 years). 4. Therefore, Ram's age = 10 + 3 = 13 years.

--- FINAL ANSWER ---

13
```