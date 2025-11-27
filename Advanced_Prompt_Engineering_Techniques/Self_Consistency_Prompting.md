# **Self Consistency Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Self-Consistency Prompting is a decoding strategy that improves multi-step reasoning by *letting a model explore multiple different reasoning paths* and then picking the answer that appears most consistently across those paths. Instead of trusting a single chain-of-thought (the usual greedy decode), self-consistency asks the model to sample many possible chains-of-thought and then take a majority vote over the final answers.

Instead of asking the AI to guess the answer once (where it might make a silly mistake), you ask it to solve the same problem multiple times using different reasoning paths. Then, you look at all the answers and pick the one that appears most frequently (a "majority vote").

![Self Consistency prompting](images/1-self-consistency-prompt.jpg)

Figure from [Self Consistency prompting](https://arxiv.org/abs/2203.11171) paper. 

## **Prompt Template**

Here is the prompt template for self consistency prompting.

```
You are a step-by-step reasoning assistant.

Use deliberate, step-by-step reasoning.

Question: {question}

Instruction:
- Think through the problem step by step.
- Produce a full chain of thought.
- Then give ONLY the final numeric answer.

Important:
- reasoning_chain must contain multiple reasoning steps.
- answer must contain ONLY the final numeric answer.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Zero-Shot Implementation**

Now let's see the implementation of zero-shot self consistency promtping technique using LangChain v1.0

```python
# !pip install langchain langchain-google-genai pydantic

import os
import time
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from collections import Counter

# ---------------------------------------------------------
# 1. Set your Gemini API key
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ---------------------------------------------------------
# 2. Define structured output model
# ---------------------------------------------------------
class SCResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Full reasoning steps")
    answer: str = Field(..., description="Final numeric answer only")


# ---------------------------------------------------------
# 3. Create parser
# ---------------------------------------------------------
parser = PydanticOutputParser(pydantic_object=SCResponse)


# ---------------------------------------------------------
# 4. Initialize Gemini model with sampling enabled
# ---------------------------------------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.8,
    top_k=40,
)


# ---------------------------------------------------------
# 5. Zero-shot Self-Consistency Prompt
# ---------------------------------------------------------
prompt_template = ChatPromptTemplate.from_template(
    """
You are a step-by-step reasoning assistant.

Use deliberate, step-by-step reasoning.

Question: {question}

Instruction:
- Think through the problem step by step.
- Produce a full chain of thought.
- Then give ONLY the final numeric answer.

Return your output in this JSON format:
{format_instructions}

Important:
- reasoning_chain must contain multiple reasoning steps.
- answer must contain ONLY the final numeric answer.
"""
)

prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())


# ---------------------------------------------------------
# 6. Build LCEL chain
# ---------------------------------------------------------
chain = prompt | model | parser


# ---------------------------------------------------------
# 7. Self-Consistency Sampling (with sleep + 5 samples)
# ---------------------------------------------------------
def self_consistency(question: str, samples: int = 5):
    answers = []
    all_outputs = []

    for i in range(samples):
        result = chain.invoke({"question": question})
        answers.append(result.answer)
        all_outputs.append(result)

        time.sleep(1)   # <-- prevents rate-limits

    final_answer = Counter(answers).most_common(1)[0][0]
    return final_answer, all_outputs


# ---------------------------------------------------------
# 8. Run on your example
# ---------------------------------------------------------
question = (
    "When I was 6 years old, my sister was half my age. Now I am 70 years old. How old is my sister?"
)

final_answer, outputs = self_consistency(question, samples=5)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------
print("\n===== SELF CONSISTENCY OUTPUT =====")
print("Final Aggregated Answer:", final_answer)

print("\n===== ALL SAMPLED REASONING PATHS =====")
for i, out in enumerate(outputs, 1):
    print(f"\n--- Sample {i} ---")
    print(out.reasoning_chain)
    print("Answer:", out.answer)
```

Here the output is
```
===== SELF CONSISTENCY OUTPUT =====
Final Aggregated Answer: 67

===== ALL SAMPLED REASONING PATHS =====

--- Sample 1 ---
First, I need to determine the age of the sister when the person was 6 years old. The problem states that when the person was 6, their sister was half their age. So, the sister's age was 6 / 2 = 3 years old. Next, I need to calculate the age difference between the person and their sister. When the person was 6 and the sister was 3, the age difference was 6 - 3 = 3 years. This age difference remains constant throughout their lives. Finally, I will apply this age difference to the person's current age. The person is now 70 years old. Since the sister is always 3 years younger, her current age is 70 - 3 = 67 years old.
Answer: 67

--- Sample 2 ---
First, I need to determine the age difference between the person and their sister. When the person was 6 years old, their sister was half their age, which means the sister was 6 / 2 = 3 years old. The age difference between them is 6 - 3 = 3 years. This age difference remains constant throughout their lives. Now, the person is 70 years old. To find the sister's current age, I subtract the constant age difference from the person's current age: 70 - 3 = 67 years old.
Answer: 67

--- Sample 3 ---
Step 1: Determine the sister's age when the person was 6 years old. The problem states the sister was half the person's age. So, sister's age = 6 / 2 = 3 years old.
Step 2: Calculate the age difference between the person and their sister. Age difference = Person's age - Sister's age = 6 - 3 = 3 years. This age difference remains constant throughout their lives.
Step 3: Apply the constant age difference to the person's current age. The person is now 70 years old. Since the sister is always 3 years younger, her current age will be 70 - 3 = 67 years old.
Answer: 67

--- Sample 4 ---
First, I need to determine the sister's age when the speaker was 6 years old. The problem states that at that time, the sister was half the speaker's age. So, 6 years / 2 = 3 years old. Next, I need to find the constant age difference between the speaker and the sister. Since the speaker was 6 and the sister was 3, the age difference is 6 - 3 = 3 years. This age difference remains constant throughout their lives. Finally, I apply this age difference to the speaker's current age. The speaker is now 70 years old. Therefore, the sister's age will be 70 - 3 = 67 years old.
Answer: 67

--- Sample 5 ---
Step 1: Determine the sister's age when the person was 6 years old. The problem states that when the person was 6, the sister was half their age. So, sister's age = 6 / 2 = 3 years old. 
Step 2: Calculate the age difference between the person and their sister. Age difference = Person's age - Sister's age = 6 - 3 = 3 years. 
Step 3: Understand that the age difference between two people remains constant over time. If the person is 3 years older than their sister, this difference will always be 3 years, regardless of how many years pass. 
Step 4: Apply the constant age difference to the person's current age. The person is now 70 years old. Since the sister is 3 years younger, her current age will be 70 - 3 = 67 years old.

Answer: 67
```


## **Few-Shot Implementation**

Now let's see the implementation of few-shot self consistency promtping technique using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
import time
from google.colab import userdata
from collections import Counter
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ---------------------------------------------------------
# 1. Set Gemini API key
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ---------------------------------------------------------
# 2. Define structured output schema
# ---------------------------------------------------------
class SCResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Full step-by-step reasoning")
    answer: str = Field(..., description="Final numeric answer only")


parser = PydanticOutputParser(pydantic_object=SCResponse)


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
# 4. Few-shot example (your earlier example)
# ---------------------------------------------------------
few_shot_example = """
Example Problem:
When I was 6 years old, my sister was half my age. Now I am 70 years old. How old is my sister?

Example Chain-of-Thought:
When I was 6, my sister was half my age, meaning she was 3. So she is always 3 years younger than me.
Now I am 70, so she must be 70 - 3 = 67.

Example Final Answer:
67
"""


# ---------------------------------------------------------
# 5. Create Few-shot Prompt Template
# ---------------------------------------------------------
prompt_template = ChatPromptTemplate.from_template(
    """
You are a step-by-step reasoning assistant.

Below is a worked example:
{few_shot_example}

Now use a similar style of reasoning to answer the new question.

New Question:
{question}

Instructions:
- Provide a full chain-of-thought reasoning.
- Then give ONLY the final numeric answer.
- Respond in this JSON format:
{format_instructions}

Important:
- reasoning_chain must contain multiple reasoning steps.
- answer must contain ONLY the final numeric answer.
"""
)

prompt = prompt_template.partial(
    format_instructions=parser.get_format_instructions(),
    few_shot_example=few_shot_example
)


# ---------------------------------------------------------
# 6. Build LCEL chain
# ---------------------------------------------------------
chain = prompt | model | parser


# ---------------------------------------------------------
# 7. Self-consistency Sampling (n_samples = 3)
# ---------------------------------------------------------
def self_consistency(question: str, samples: int = 3):
    answers = []
    outputs = []

    for _ in range(samples):
        result = chain.invoke({"question": question})
        outputs.append(result)
        answers.append(result.answer)

        time.sleep(1)   # Avoid rate-limit issues

    final_answer = Counter(answers).most_common(1)[0][0]
    return final_answer, outputs


# ---------------------------------------------------------
# 8. Run Few-shot Self-Consistency
# ---------------------------------------------------------
question = "If it takes 1 hour to dry 3 shirts outside on a sunny line, how long does it take to dry 9 shirts?"

final_answer, samples = self_consistency(question, samples=3)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------
print("\n===== FINAL AGGREGATED ANSWER =====")
print(final_answer)

print("\n===== ALL SAMPLED REASONING PATHS =====")
for i, out in enumerate(samples, 1):
    print(f"\n--- Sample {i} ---")
    print(out.reasoning_chain)
    print("Answer:", out.answer)
```

Here the output is
```
===== FINAL AGGREGATED ANSWER =====
1

===== ALL SAMPLED REASONING PATHS =====

--- Sample 1 ---
The key factor in drying shirts outside on a sunny line is the time it takes for the sun and air to dry the fabric. All shirts placed on the line at the same time will dry simultaneously. If it takes 1 hour for 3 shirts to dry, it means that each individual shirt dries in 1 hour. Therefore, if you place 9 shirts on the line at the same time (assuming sufficient space and sun), they will all dry simultaneously, and the total time required will still be 1 hour.
Answer: 1

--- Sample 2 ---
The problem states that it takes 1 hour to dry 3 shirts outside on a sunny line. When shirts are hung on a line, they dry simultaneously, assuming they are all exposed to the same conditions (sun, wind). The drying time is determined by the environmental factors, not by the number of items drying at the same time, as long as there is enough space. Therefore, if 3 shirts dry in 1 hour, each individual shirt takes 1 hour to dry. If you hang 9 shirts on the line at the same time, they will all be drying concurrently under the same conditions. Consequently, it will still take 1 hour for all 9 shirts to dry.
Answer: 1

--- Sample 3 ---
The problem states that it takes 1 hour to dry 3 shirts. When drying shirts on a line outside, the shirts dry simultaneously, not sequentially. This means that if you put 3 shirts out, they all dry within that 1 hour. If you put 9 shirts out, assuming there is enough space on the line for all of them to be exposed to the sun and air at the same time, they will all be drying at the same rate. Therefore, the total time required for all 9 shirts to dry will still be the same amount of time it takes for one shirt (or any number of shirts placed simultaneously) to dry under those conditions.

Answer: 1
```