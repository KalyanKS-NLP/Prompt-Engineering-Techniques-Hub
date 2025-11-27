# **Faithful Chain of Thought Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates. 

## **Overview**
Faithful Chain-of-Thought (CoT) prompting is a technique for Large Language Models (LLMs) that ensures the final answer is directly and logically derived from the step-by-step reasoning the model provides. It works in two key stages:
1. Translation 
- The LLM takes the original question (in natural language, like English) and translates it into a precise, step-by-step plan called a reasoning chain.
- This reasoning chain is a mix of natural language comments (for human understanding) and symbolic language (like computer code, e.g., Python, or a formal planning language).

2. Problem Solving
- The reasoning chain, which now includes executable code, is then handed off to a deterministic solver (like a Python interpreter or calculator program).
- The solver executes the steps in the chain to mathematically or logically calculate the final answer.

By embedding both *planning* (natural language decomposition) and *computation* (symbolic code), Faithful CoT produces explanations that are both interpretable and reliable.

![Faithful CoT prompting](images/4-faithful-cot-prompt.jpg)

Figure from [Faithful CoT prompting](https://arxiv.org/abs/2301.13379) paper.

## **Prompt Template**

Here is the prompt template for program of thoughts 

```
You are an expert reasoning assistant.

Use Faithful Chain-of-Thought (Faithful CoT) prompting.

You must output a single Python program that:

- Includes brief natural-language substeps as comments.
- Uses Python variable assignments for each step.
- Computes the final answer deterministically.
- MUST end with: ans = <final value>
- MUST be executable as-is in a Python interpreter.

Problem:
{question}
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Zero-Shot Implementation**

Now let's see the implementation of zero-shot faithful CoT promtping technique using LangChain v1.0

```python
# !pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_experimental.utilities import PythonREPL

# 1. Set Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# 2. Faithful CoT Structured Output Format
class FaithfulCoTResponse(BaseModel):
    program: str = Field(
        ..., 
        description="Faithful chain-of-thought reasoning in Python. May include comments. Must end with ans = <final value>."
    )

# 3. Parser
parser = PydanticOutputParser(pydantic_object=FaithfulCoTResponse)

# 4. Initialize Gemini model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Python REPL
python_repl = PythonREPL()

# 6. Zero-Shot Faithful CoT Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert reasoning assistant.

Use **Faithful Chain-of-Thought (Faithful CoT)** prompting.

You must output a single Python program that:

- Includes brief natural-language substeps as comments.
- Uses Python variable assignments for each step.
- Computes the final answer deterministically.
- MUST end with: ans = <final value>
- MUST be executable as-is in a Python interpreter.

The output MUST be a JSON object containing only one field: "program".

Problem:
{question}

Provide the solution in this JSON format:
{format_instructions}
"""
)

# 7. Insert parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 8. Build chain
chain = prompt | model | parser

# 9. Use your earlier example
question = """
Daniel has 17 apples. Rosy gives Daniel 5 oranges, and in return,
Daniel gives her 3 apples. How many apples does Daniel have now?
"""

# 10. Invoke LLM
result = chain.invoke({"question": question})

print("\n--- Faithful Chain-of-Thought Program ---\n")
print(result.program)

# 11. Execute program
execution_output = python_repl.run(result.program)

# 12. Extract final answer
final_answer = python_repl.locals.get("ans", None)

print("\n--- Final Answer (from Python interpreter) ---\n")
print(final_answer)
```

Here the output is
```
# Daniel's initial number of apples.
initial_apples = 17

# Rosy gives Daniel oranges, which does not change the number of apples Daniel has.
# oranges_given_to_daniel = 5

# Daniel gives Rosy some apples.
apples_given_to_rosy = 3

# Calculate the number of apples Daniel has after giving some to Rosy.
apples_after_giving_to_rosy = initial_apples - apples_given_to_rosy

# The final number of apples Daniel has.
ans = apples_after_giving_to_rosy

--- Final Answer (from Python interpreter) ---

14
```


## **Few-Shot Implementation**

Now let's see the implementation of few-shot faithful CoT promtping technique using LangChain v1.0

```python

# pip install langchain langchain-google-genai pydantic langchain-experimental

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_experimental.utilities import PythonREPL

# 1. Set API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# 2. Structured Faithful CoT Response
class FaithfulCoTResponse(BaseModel):
    program: str = Field(
        ...,
        description="Executable Python program containing faithful chain-of-thought with comments. Must assign final value to 'ans'."
    )

# 3. Parser
parser = PydanticOutputParser(pydantic_object=FaithfulCoTResponse)

# 4. Initialize Gemini model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Python Interpreter (can run code + comments)
python_repl = PythonREPL()

# 6. FEW-SHOT EXAMPLE (Your previous Faithful CoT example)
few_shot_example = """
Question: Daniel has 17 apples. Rosy gives Daniel 5 oranges, and in return Daniel gives her 3 apples. How many apples does Daniel have now?

# 1. How many apples does Daniel begin with?
n_apples_begin = 17

# 2. How many apples does Daniel give away?
n_apples_given = 3

# 3. Final apples Daniel has now
n_apples_final = n_apples_begin - n_apples_given

ans = n_apples_final
"""

# 7. Few-Shot Faithful CoT Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert reasoning assistant.

Below is an example demonstrating **Faithful Chain-of-Thought (Faithful CoT) prompting**.
The solution is expressed as Python code with natural language reasoning embedded as comments.

The final answer MUST be assigned to `ans`.

{few_shot_example}

Now solve the following problem using the SAME Faithful CoT format:

Problem:
{question}

Output Instructions:
- Output ONLY a single Python program.
- Comments ARE allowed.
- Code must be fully executable.
- Must end with: ans = <final value>

Return the output in this JSON format:
{format_instructions}
"""
)

# 8. Insert few-shot example + parser instructions
prompt = prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

# 9. Build chain
chain = prompt | model | parser

# 10. Current Problem
question = """
A bakery starts the day with 40 croissants. They sell 25 croissants in the morning and 12 in the afternoon.
If they want to have at least 5 croissants left for the evening staff, do they meet this target?
"""

# 11. Invoke LLM → generate Faithful CoT program
result = chain.invoke({"question": question})

print("\n--- Faithful CoT Program Generated by LLM ---\n")
print(result.program)

# 12. Execute program using the Python REPL
execution_output = python_repl.run(result.program)

# 13. Retrieve final answer
final_answer = python_repl.locals.get("ans", None)

print("\n--- Final Answer (from Python interpreter) ---\n")
print(final_answer)
```

Here the output is
```
--- Faithful CoT Program Generated by LLM ---

# 1. How many croissants did the bakery start with?
croissants_start = 40

# 2. How many croissants were sold in the morning?
croissants_sold_morning = 25

# 3. How many croissants were sold in the afternoon?
croissants_sold_afternoon = 12

# 4. Calculate the total number of croissants sold.
total_croissants_sold = croissants_sold_morning + croissants_sold_afternoon

# 5. Calculate the number of croissants remaining after sales.
croissants_remaining = croissants_start - total_croissants_sold

# 6. Determine the target number of croissants for the evening staff.
target_croissants_evening = 5

# 7. Check if the bakery meets the target (at least 5 croissants left).
ans = croissants_remaining >= target_croissants_evening

--- Final Answer (from Python interpreter) ---

False
```
