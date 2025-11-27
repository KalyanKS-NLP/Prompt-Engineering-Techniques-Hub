# **Least to Most Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Least-to-Most Prompting (LtM) is a prompting technique in which a complex question is solved by:

1. Decomposing it into simpler, smaller sub-problems (the “least” stage)
2. Solving each sub-problem sequentially and using each solution to build toward the final answer (the “most” stage)

Instead of tackling the entire problem at once, the model breaks it down into manageable steps, solves each step in order, and gradually works toward the final solution. This approach mirrors how humans solve complex tasks: first understand the parts, then combine the answers.

Unlike Chain-of-Thought (CoT), which focuses on generating one long reasoning path, Least-to-Most prompting explicitly separates decomposition and solution. It forces the model to *plan first* and *compute later*, enabling stronger reasoning, especially for multi-stage problems.

![Least to Most prompting](images/1-least-prompt.jpg)

Figure from [Least to Most prompting](https://arxiv.org/abs/2205.10625) paper. 


## **Prompt Template**

Here is the prompt template for least to most prompting.
```
You are an expert reasoning assistant.

You must solve the problem using **Least-to-Most Prompting**, which has TWO required stages:

1. **Decomposition (Least):**
   - Break the main problem into a sequential list of simpler sub-problems.

2. **Sequential Solving (Most):**
   - Solve each sub-problem step-by-step.
   - Use outputs of earlier sub-problems to solve later ones.
   - Continue until the final answer is reached.

Question:
{question}

Important:
- decomposition must contain numbered sub-problems.
- sequential_solution must show calculations for each sub-problem.
- final_answer must contain ONLY the final numeric answer.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Zero-Shot Implementation**

Now let's see the implementation of zero-shot least to most promtping technique using LangChain v1.0

```python
# !pip install langchain langchain-google-genai pydantic

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# 2. Define structured output for LtM
class LtMResponse(BaseModel):
    decomposition: str = Field(..., description="List of sub-problems in order")
    sequential_solution: str = Field(..., description="Step-by-step solutions for each sub-problem")
    final_answer: str = Field(..., description="Final numeric answer only")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=LtMResponse)

# 4. Initialize Gemini model (gemini-2.5-flash)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google",
    temperature=0
)

# 5. Zero-Shot Least-to-Most Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert reasoning assistant.

You must solve the problem using **Least-to-Most Prompting**, which has TWO required stages:

1. **Decomposition (Least):**
   - Break the main problem into a sequential list of simpler sub-problems.

2. **Sequential Solving (Most):**
   - Solve each sub-problem step-by-step.
   - Use outputs of earlier sub-problems to solve later ones.
   - Continue until the final answer is reached.

Question:
{question}

Provide your solution in the following JSON format:
{format_instructions}

Important:
- decomposition must contain numbered sub-problems.
- sequential_solution must show calculations for each sub-problem.
- final_answer must contain ONLY the final numeric answer.
"""
)

# 6. Insert parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build LCEL chain
chain = prompt | model | parser

# 8. Invoke the chain using your marathon LtM problem
question = """
A runner is preparing for a marathon. She runs 10 miles every day.
Last week, she ran 7 days.
This week, she took a 2-day rest and ran 8 miles on the remaining days.
If she wants to run a total of 180 miles across both weeks,
how many more miles must she run in the next 3 days?
"""

result = chain.invoke({"question": question})

# 9. Output
print(result)
print("\n--- Decomposition ---\n", result.decomposition)
print("\n--- Sequential Solution ---\n", result.sequential_solution)
print("\n--- Final Answer ---\n", result.final_answer)

```

Here the output is
```
--- Decomposition ---
 1. Calculate the total miles run last week.
2. Calculate the number of days the runner ran this week.
3. Calculate the total miles run this week.
4. Calculate the total miles run in both weeks combined.
5. Calculate the remaining miles needed to reach the target of 180 miles.

--- Sequential Solution ---
 1. **Total miles run last week:**
   She ran 10 miles/day for 7 days.
   Miles last week = 10 miles/day * 7 days = 70 miles.

2. **Number of days the runner ran this week:**
   A week has 7 days. She took a 2-day rest.
   Days ran this week = 7 days - 2 days = 5 days.

3. **Total miles run this week:**
   She ran 8 miles on the 5 days she ran this week.
   Miles this week = 8 miles/day * 5 days = 40 miles.

4. **Total miles run in both weeks combined:**
   Total miles so far = Miles last week + Miles this week
   Total miles so far = 70 miles + 40 miles = 110 miles.

5. **Remaining miles needed to reach the target of 180 miles:**
   Target total miles = 180 miles.
   Remaining miles = Target total miles - Total miles so far
   Remaining miles = 180 miles - 110 miles = 70 miles.

   Therefore, she must run 70 more miles in the next 3 days.

--- Final Answer ---
 70
```
## **Few-Shot Implementation**

Now let's see the implementation of few-shot least to most promtping technique using LangChain v1.0

```python

# !pip install langchain langchain-google-genai pydantic

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# 2. Define Pydantic schema for LtM output
class LtMResponse(BaseModel):
    decomposition: str = Field(..., description="Ordered list of sub-problems")
    sequential_solution: str = Field(..., description="Step-by-step reasoning for each sub-problem")
    final_answer: str = Field(..., description="Final numeric answer only")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=LtMResponse)

# 4. Initialize Gemini model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Few-shot LtM example (1-shot)
few_shot_example = """
Goal: Solve the problem using Least-to-Most prompting.

Problem:
A runner is preparing for a marathon. She runs 10 miles every day.
Last week, she ran 7 days. This week, she took a 2-day rest and
ran 8 miles on the remaining days. If she wants to run a total of
180 miles across both weeks, how many more miles must she run in the next 3 days?

1. Decomposition (Least):
- Sub-problem 1: Calculate total miles run last week.
- Sub-problem 2: Calculate number of running days this week.
- Sub-problem 3: Calculate total miles run this week.
- Sub-problem 4: Calculate total miles run so far.
- Sub-problem 5: Calculate remaining miles needed.

2. Sequential Solving (Most):
- Sub-problem 1: 10 miles/day × 7 days = 70 miles
- Sub-problem 2: 7 days − 2 rest days = 5 days
- Sub-problem 3: 8 miles/day × 5 days = 40 miles
- Sub-problem 4: 70 + 40 = 110 miles
- Sub-problem 5: 180 − 110 = 70 miles

Final Answer: 70
"""

# 6. Few-shot LtM prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert reasoning assistant.

Below is an example problem solved using **Least-to-Most prompting**:
{few_shot_example}

Now apply the same Least-to-Most structure to solve the following problem:

Question: {question}

Provide the answer in the following JSON format:
{format_instructions}
"""
)

# 7. Inject the few-shot example + parser instructions
prompt = prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

# 8. Build LCEL chain
chain = prompt | model | parser

# 9. Target problem (Chef question)
question = (
    "A chef needs to make 30 croissants. It takes him 5 minutes to prepare "
    "the dough for one croissant, and 15 minutes to bake it. He has already "
    "prepared and baked 5 croissants. He has 4 hours remaining. How many more "
    "minutes does he have left after preparing and baking the rest of the "
    "required croissants?"
)

# 10. Run the chain
result = chain.invoke({"question": question})

# 11. Display result
print("\n--- Decomposition ---\n", result.decomposition)
print("\n--- Sequential Solution ---\n", result.sequential_solution)
print("\n--- Final Answer ---\n", result.final_answer)
```

Here the output is
```
--- Decomposition ---
 - Sub-problem 1: Calculate the number of croissants remaining to be made.
- Sub-problem 2: Calculate the total time (preparation + baking) required for one croissant.
- Sub-problem 3: Calculate the total time needed to prepare and bake the remaining croissants.
- Sub-problem 4: Convert the chef's total remaining time (4 hours) into minutes.
- Sub-problem 5: Calculate the minutes the chef has left after completing the remaining croissants.

--- Sequential Solution ---
 - Sub-problem 1: 30 total croissants - 5 already made = 25 croissants remaining.
- Sub-problem 2: 5 minutes (prepare) + 15 minutes (bake) = 20 minutes per croissant.
- Sub-problem 3: 25 remaining croissants × 20 minutes/croissant = 500 minutes needed.
- Sub-problem 4: 4 hours × 60 minutes/hour = 240 minutes available.
- Sub-problem 5: 240 minutes (available) - 500 minutes (needed) = -260 minutes.

--- Final Answer ---
 -260
```