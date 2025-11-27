# **Step Back Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates. 

## **Overview**

Step-Back Prompting is a reasoning technique that improves problem-solving by encouraging the model to *temporarily step back* from the specific question and reflect on the more general principle that governs the solution.

Instead of jumping directly into computations, the model is first guided to identify the high-level concept or first-principle law relevant to the task. Once this abstraction is established, the model uses that principle to reason clearly and arrive at the final answer.

This two-stage process, *Abstraction → Reasoning* helps the model avoid errors caused by focusing too narrowly on surface details. By anchoring the solution to a general principle, Step-Back Prompting improves accuracy, structure, and conceptual grounding.

![Step Back prompting](images/2-step-back-prompt.jpg)

Figure from [Step Back prompting](https://arxiv.org/abs/2311.04205) paper.


##  **Prompt Template**

Here is the step back abstraction prompt template for step back prompting.

```
You are an expert in abstraction.

Given the original question below:

Original Question:
{question}

Perform TWO tasks:
1. Generate a high-level step-back question that captures the general principle needed.
2. Answer that step-back question by giving the underlying principle or formula.
```

Here is the final reasoning prompt template for step back prompting.
```
You are an expert problem solver.

Use the abstract principle retrieved earlier to answer the original question.

Original Question:
{question}

Step-Back Principle:
{abstraction}

Now solve the original question step by step.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week


## **Implementation**

Now let's see the implementation of step back promtping technique using LangChain v1.0

```python
# !pip install langchain langchain-google-genai pydantic

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
# 2. Define Structured Output Models
# ----------------------------------------------------------

class Abstraction(BaseModel):
    stepback_question: str = Field(..., description="The abstract step-back question")
    stepback_answer: str = Field(..., description="The high-level principle that answers the step-back question")


class FinalAnswer(BaseModel):
    final_answer: str = Field(..., description="The final solution using the abstract principle")


abstraction_parser = PydanticOutputParser(pydantic_object=Abstraction)
final_answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)


# ----------------------------------------------------------
# 3. Initialize Gemini model
# ----------------------------------------------------------

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)


# ----------------------------------------------------------
# 4. Prompt Templates (ONLY TWO CALLS)
# ----------------------------------------------------------

# --- Call 1: Step-Back Abstraction ---
abstraction_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert in abstraction.

Given the original question below:

Original Question:
{question}

Perform TWO tasks:
1. Generate a high-level **step-back question** that captures the general principle needed.
2. Answer that step-back question by giving the **underlying principle or formula**.

Return BOTH in this JSON format:
{format_instructions}
"""
)

abstraction_prompt = abstraction_prompt_template.partial(
    format_instructions=abstraction_parser.get_format_instructions()
)


# --- Call 2: Final Reasoning ---
final_reasoning_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert problem solver.

Use the abstract principle retrieved earlier to answer the original question.

Original Question:
{question}

Step-Back Principle:
{abstraction}

Now solve the original question step by step.

Return the final answer in this JSON format:
{format_instructions}
"""
)

final_reasoning_prompt = final_reasoning_prompt_template.partial(
    format_instructions=final_answer_parser.get_format_instructions()
)


# ----------------------------------------------------------
# 5. Build LCEL Chains (Only Two Calls)
# ----------------------------------------------------------

abstraction_chain = abstraction_prompt | model | abstraction_parser
final_answer_chain = final_reasoning_prompt | model | final_answer_parser


# ----------------------------------------------------------
# 6. Run Step-Back Prompting on Your Example
# ----------------------------------------------------------

question = "A train travels at 60 miles per hour. How far will it travel in 3 hours?"


# Call 1 — Abstraction
abs_result = abstraction_chain.invoke({"question": question})
print("\n--- STEP-BACK ABSTRACTION ---\n")
print("Step-Back Question:", abs_result.stepback_question)
print("Step-Back Answer:", abs_result.stepback_answer)


# Call 2 — Reasoning
final_result = final_answer_chain.invoke({
    "question": question,
    "abstraction": abs_result.stepback_answer
})
print("\n--- FINAL ANSWER ---\n")
print(final_result.final_answer)
```

Here the output is
```
--- STEP-BACK ABSTRACTION ---

Step-Back Question: How is the total distance covered related to the constant rate of travel and the duration of that travel?
Step-Back Answer: The total distance traveled is calculated by multiplying the constant rate of travel (speed) by the time spent traveling. This relationship is commonly expressed by the formula: Distance = Rate × Time (D = R × T).

--- FINAL ANSWER ---

To find the distance, we use the formula Distance = Rate × Time. Given the rate (speed) is 60 miles per hour and the time is 3 hours, we multiply 60 mph by 3 hours. Distance = 60 miles/hour × 3 hours = 180 miles. Therefore, the train will travel 180 miles in 3 hours.
```
