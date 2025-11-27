# **Chain of Verification Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Chain-of-Verification (CoVe) Prompting is a structured reasoning technique designed to reduce factual errors (hallucinations) by forcing a model to *verify its own answer* before finalizing it.

Unlike standard Chain-of-Thought—where the model directly reasons and produces a , CoVe breaks the process into four deliberate stages:

1. Draft an initial answer
2. Generate verification questions that check the facts in the draft
3. Independently answer each verification question without seeing the draft
4. Produce a corrected, final answer based on the verified facts

This separation into *“draft → verify → correct”* helps prevent the model from repeating its own mistakes.

By treating verification as an independent fact-checking phase, CoVe significantly reduces hallucinations, especially in factual or knowledge-based tasks.


![Chain of Verification prompting](images/2-cove-prompt.jpg)

Figure from [Chain of Verification prompting](https://arxiv.org/abs/2309.11495) paper.

## **Prompt Template**

Here is the  draft prompt template for chain of verification prompting.

```
You are a factual answering assistant.

Step 1 of Chain-of-Verification:
Generate a baseline draft answer for the question. Do NOT verify anything yet.

Question:
{question}
```

Here is the  verification questions generation prompt template for chain of verification prompting.
```
You are now performing Step 2 of Chain-of-Verification.

Given the baseline draft answer below, generate verification questions to check EACH factual claim.

Draft Answer:
{draft}

Your job:
- Break the draft into factual claims.
- Create one verification question for each claim.
- Each question MUST be independently fact-checkable.
```

Here is the  verification answers generation prompt template for chain of verification prompting.
```
Step 3 of Chain-of-Verification.

Answer the following verification questions INDEPENDENTLY.
Do NOT refer to the draft answer. Use only factual knowledge.

Questions:
{questions}
```

Here is the  final response generation prompt template for chain of verification prompting.
```
Step 4 of Chain-of-Verification.

You are given:
1. The baseline draft answer
2. The list of verification questions
3. The factual answers to those questions

Your task:
- Identify incorrect statements in the draft
- Keep only the claims supported by verification answers
- Remove or correct unsupported claims
- Produce the final VERIFIED answer

Draft:
{draft}

Verification Questions:
{questions}

Verification Answers:
{answers}
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of chain of verification prompting technique using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
import time 
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 1. Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# ---------------------------------------------------------
# 2. Define structured outputs for all 4 CoVe stages
# ---------------------------------------------------------

class BaselineResponse(BaseModel):
    draft_answer: str = Field(..., description="Initial unverified answer")

class VerificationPlan(BaseModel):
    questions: List[str] = Field(..., description="List of verification questions generated from the draft")

class VerificationAnswers(BaseModel):
    answers: List[str] = Field(..., description="Answers to the verification questions in the same order")

class FinalVerifiedResponse(BaseModel):
    verified_answer: str = Field(..., description="Final corrected answer using only verified facts")

# ---------------------------------------------------------
# 3. Initialize the Gemini model (gemini-2.5-flash)
# ---------------------------------------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# ---------------------------------------------------------
# 4. PROMPTS
# ---------------------------------------------------------

# 4.1 Baseline Draft Prompt
baseline_prompt_tmpl = ChatPromptTemplate.from_template(
    """
You are a factual answering assistant.

Step 1 of Chain-of-Verification:
Generate a baseline draft answer for the question. Do NOT verify anything yet.

Question:
{question}

Return your response in JSON:
{format_instructions}
"""
)

baseline_parser = PydanticOutputParser(pydantic_object=BaselineResponse)
baseline_prompt = baseline_prompt_tmpl.partial(format_instructions=baseline_parser.get_format_instructions())


# 4.2 Plan Verification Questions Prompt
verify_plan_tmpl = ChatPromptTemplate.from_template(
    """
You are now performing Step 2 of Chain-of-Verification.

Given the baseline draft answer below, generate verification questions to check EACH factual claim.

Draft Answer:
{draft}

Your job:
- Break the draft into factual claims.
- Create one verification question for each claim.
- Each question MUST be independently fact-checkable.

Return JSON:
{format_instructions}
"""
)

verify_plan_parser = PydanticOutputParser(pydantic_object=VerificationPlan)
verify_plan_prompt = verify_plan_tmpl.partial(format_instructions=verify_plan_parser.get_format_instructions())


# 4.3 Verification Answering Prompt
verify_answer_tmpl = ChatPromptTemplate.from_template(
    """
Step 3 of Chain-of-Verification.

Answer the following verification questions INDEPENDENTLY.
Do NOT refer to the draft answer. Use only factual knowledge.

Questions:
{questions}

Return JSON:
{format_instructions}
"""
)

verify_answer_parser = PydanticOutputParser(pydantic_object=VerificationAnswers)
verify_answer_prompt = verify_answer_tmpl.partial(format_instructions=verify_answer_parser.get_format_instructions())


# 4.4 Final Verified Response Prompt
final_answer_tmpl = ChatPromptTemplate.from_template(
    """
Step 4 of Chain-of-Verification.

You are given:
1. The baseline draft answer
2. The list of verification questions
3. The factual answers to those questions

Your task:
- Identify incorrect statements in the draft
- Keep only the claims supported by verification answers
- Remove or correct unsupported claims
- Produce the final VERIFIED answer

Draft:
{draft}

Verification Questions:
{questions}

Verification Answers:
{answers}

Return JSON:
{format_instructions}
"""
)

final_answer_parser = PydanticOutputParser(pydantic_object=FinalVerifiedResponse)
final_answer_prompt = final_answer_tmpl.partial(format_instructions=final_answer_parser.get_format_instructions())

# ---------------------------------------------------------
# 5. Build the LCEL chains
# ---------------------------------------------------------

baseline_chain = baseline_prompt | model | baseline_parser
time.sleep(1)
plan_chain = verify_plan_prompt | model | verify_plan_parser
time.sleep(1)
verify_chain = verify_answer_prompt | model | verify_answer_parser
time.sleep(1)
final_chain = final_answer_prompt | model | final_answer_parser

# ---------------------------------------------------------
# 6. Run CoVe on your example
# ---------------------------------------------------------

question = "Which US Presidents were born in the state of Texas?"

# Step 1: Baseline Draft
baseline = baseline_chain.invoke({"question": question})

# Step 2: Plan Verifications
plan = plan_chain.invoke({"draft": baseline.draft_answer})

# Step 3: Execute Verifications
verification = verify_chain.invoke({"questions": plan.questions})

# Step 4: Final Verified Answer
final = final_chain.invoke({
    "draft": baseline.draft_answer,
    "questions": plan.questions,
    "answers": verification.answers
})

# ---------------------------------------------------------
# 7. Print all outputs
# ---------------------------------------------------------

print("\n--- Baseline Draft ---\n", baseline.draft_answer)
print("\n--- Verification Questions ---\n", plan.questions)
print("\n--- Verification Answers ---\n", verification.answers)
print("\n--- Final Verified Answer ---\n", final.verified_answer)
```

Here the output is
```
--- Baseline Draft ---
 The US Presidents born in the state of Texas are Lyndon B. Johnson and Dwight D. Eisenhower.

--- Verification Questions ---
 ['Was Lyndon B. Johnson a US President?', 'Was Lyndon B. Johnson born in the state of Texas?', 'Was Dwight D. Eisenhower a US President?', 'Was Dwight D. Eisenhower born in the state of Texas?', 'Are there any other US Presidents, besides Lyndon B. Johnson and Dwight D. Eisenhower, who were born in the state of Texas?']

--- Verification Answers ---
 ['Yes, Lyndon B. Johnson was the 36th President of the United States.', 'Yes, Lyndon B. Johnson was born near Stonewall, Texas.', 'Yes, Dwight D. Eisenhower was the 34th President of the United States.', 'Yes, Dwight D. Eisenhower was born in Denison, Texas.', 'No, there are no other US Presidents, besides Lyndon B. Johnson and Dwight D. Eisenhower, who were born in the state of Texas.']

--- Final Verified Answer ---
 The US Presidents born in the state of Texas are Lyndon B. Johnson and Dwight D. Eisenhower.
 ```