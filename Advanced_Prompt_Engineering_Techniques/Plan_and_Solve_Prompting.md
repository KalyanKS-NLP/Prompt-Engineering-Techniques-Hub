# **Plan and Solve Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Plan-and-Solve Prompting is a break down prompting technique that guides a model to approach complex problems in two deliberate stages:

1. First create a plan — understand the problem, extract important information, and outline the steps required to solve it.
2. Then execute the plan — compute each step carefully and derive the final answer.

![Plan and Solve prompting](images/2-plan-solve-prompt.jpg)

Figure from [Plan and Solve prompting](https://arxiv.org/abs/2305.04091) paper. 

## **Prompt Template**
Here is the prompt template for plan and solve prompting.

```
You are an expert step-by-step reasoning assistant using plan and solve prompting following the instruction.
Let’s first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. 

Then, let’s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer."

Question:
{question}

Answer: 

Important:
- variables must list each extracted variable and its numeric value.
- plan must contain a numbered plan of steps to compute the answer.
- calculation must show the step-by-step execution of the plan with arithmetic.
- final_answer must contain ONLY the final numeric answer (no units, no explanation).
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week


## **Zero-Shot Implementation**
Now let's see the implementation of zero-shot plan and solve promtping technique using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# 2. Define structured output for Plan-and-Solve
class PlanSolveResponse(BaseModel):
    variables: str = Field(..., description="Extracted relevant variables and their numerals")
    plan: str = Field(..., description="A complete step-by-step plan to solve the problem")
    calculation: str = Field(..., description="Execution of the plan with intermediate calculations")
    final_answer: str = Field(..., description="Final numeric answer only")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=PlanSolveResponse)

# 4. Initialize Gemini model (gemini-2.5-flash)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Zero-Shot Plan-and-Solve Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert step-by-step reasoning assistant using plan and solve prompting following the instruction.
Let’s first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. 
Then, let’s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), 
solve the problem step by step, and show the answer."


Question:
{question}

Answer: 

Provide your solution in the following JSON format exactly:
{format_instructions}

Important:
- variables must list each extracted variable and its numeric value.
- plan must contain a numbered plan of steps to compute the answer.
- calculation must show the step-by-step execution of the plan with arithmetic.
- final_answer must contain ONLY the final numeric answer (no units, no explanation).
"""
)

# 6. Insert parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build LCEL chain
chain = prompt | model | parser

# 8. Invoke the chain using your train-speed Plan-and-Solve problem
question = """
A train travels at an average speed of 60 mph for the first 3 hours of a journey and then at an average speed
of 40 mph for the remaining 2 hours. What is the average speed of the train for the entire journey?

Answer Choices: (A) 52 mph (B) 50 mph (C) 48 mph (D) 46 mph (E) 45 mph
"""

result = chain.invoke({"question": question})

# 9. Output
print("\n--- Variables ---\n", result.variables)
print("\n--- Plan ---\n", result.plan)
print("\n--- Calculation ---\n", result.calculation)
print("\n--- Final Answer ---\n", result.final_answer)

```

Here the output is
```
--- Variables ---
 Speed for the first part of the journey (S1) = 60 mph, Time for the first part of the journey (T1) = 3 hours, Speed for the second part of the journey (S2) = 40 mph, Time for the second part of the journey (T2) = 2 hours

--- Plan ---
 1. Calculate the distance covered in the first part of the journey (D1) using the formula D1 = S1 × T1. 2. Calculate the distance covered in the second part of the journey (D2) using the formula D2 = S2 × T2. 3. Calculate the total distance covered (D_total) by adding D1 and D2. 4. Calculate the total time taken for the journey (T_total) by adding T1 and T2. 5. Calculate the average speed for the entire journey (S_average) using the formula S_average = D_total / T_total.

--- Calculation ---
 1. Distance for the first part (D1) = 60 mph × 3 hours = 180 miles.
2. Distance for the second part (D2) = 40 mph × 2 hours = 80 miles.
3. Total distance (D_total) = D1 + D2 = 180 miles + 80 miles = 260 miles.
4. Total time (T_total) = T1 + T2 = 3 hours + 2 hours = 5 hours.
5. Average speed (S_average) = D_total / T_total = 260 miles / 5 hours = 52 mph.

--- Final Answer ---
 52
```


## **Few-Shot Implementation**
Now let's see the implementation of few-shot plan and solve promtping technique using LangChain v1.0

```python
# !pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# 2. Define Pydantic schema for Plan-and-Solve output
class PlanSolveResponse(BaseModel):
    variables: str = Field(..., description="Extracted variables with their numerical values")
    plan: str = Field(..., description="A complete numbered plan for solving the problem")
    calculation: str = Field(..., description="Execution of the plan with intermediate steps")
    final_answer: str = Field(..., description="Final numeric answer only")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=PlanSolveResponse)

# 4. Initialize Gemini model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Few-shot Plan-and-Solve example (train problem)
few_shot_example = """
Goal: Solve the problem using Plan-and-Solve prompting.

Problem:
A train travels at an average speed of 60 mph for the first 3 hours
and then at 40 mph for the next 2 hours. What is the average speed
for the entire journey? Answer Choices: (A) 52 mph (B) 50 mph (C) 48 mph (D) 46 mph (E) 45 mph

1. Variables:
- S1 = 60 mph
- T1 = 3 hours
- S2 = 40 mph
- T2 = 2 hours

2. Plan:
1. Compute D1 = S1 × T1
2. Compute D2 = S2 × T2
3. Compute total distance = D1 + D2
4. Compute total time = T1 + T2
5. Compute average speed = total distance ÷ total time

3. Calculation:
- D1 = 60 × 3 = 180 miles
- D2 = 40 × 2 = 80 miles
- Total distance = 180 + 80 = 260
- Total time = 3 + 2 = 5
- Average speed = 260 ÷ 5 = 52

Final Answer: 52
"""

# 6. Few-shot Plan-and-Solve prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert reasoning assistant.

Below is an example problem solved using **Plan-and-Solve prompting**:
{few_shot_example}

Now apply the same Plan-and-Solve structure to solve the following problem.

You must start your reasoning with this exact trigger sentence:

"Let’s first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. Then, let’s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and show the answer."

Question: {question}

Provide the answer in the following JSON format:
{format_instructions}
"""
)

# 7. Inject example + parser instructions
prompt = prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

# 8. Build LCEL chain
chain = prompt | model | parser

# 9. Target problem (chemist dilution question)
question = (
    "A chemist has 40 liters of a solution that is 30% acid. "
    "How many liters of pure water must be added to dilute the solution "
    "so that the final mixture is 10% acid? "
    "Answer Choices: (A) 60 liters (B) 70 liters (C) 80 liters "
    "(D) 90 liters (E) 100 liters"
)

# 10. Run the chain
result = chain.invoke({"question": question})

# 11. Display output sections
print("\n--- Variables ---\n", result.variables)
print("\n--- Plan ---\n", result.plan)
print("\n--- Calculation ---\n", result.calculation)
print("\n--- Final Answer ---\n", result.final_answer)

```

Here the output is
```
--- Variables ---
 V_initial = 40 liters (initial volume of solution)
C_initial = 30% = 0.30 (initial acid concentration)
C_final = 10% = 0.10 (final acid concentration)
W_added = ? (liters of pure water to be added)

--- Plan ---
 1. Calculate the initial amount of acid in the solution (Amount_acid = V_initial × C_initial).
2. Recognize that adding pure water does not change the amount of acid in the solution.
3. Define the final total volume of the solution (V_final = V_initial + W_added).
4. Set up an equation using the final acid concentration: Amount_acid = C_final × V_final.
5. Substitute the expression for V_final into the equation: Amount_acid = C_final × (V_initial + W_added).
6. Solve the equation for W_added.

--- Calculation ---
 1. Calculate initial amount of acid:
   Amount_acid = V_initial × C_initial = 40 liters × 0.30 = 12 liters
2. The amount of acid in the final solution remains 12 liters.
3. Let V_final be the total volume after adding water.
4. Set up the equation for the final concentration:
   Amount_acid = C_final × V_final
   12 = 0.10 × V_final
5. Solve for V_final:
   V_final = 12 / 0.10 = 120 liters
6. Calculate the amount of water added:
   W_added = V_final - V_initial
   W_added = 120 liters - 40 liters = 80 liters

--- Final Answer ---
 80
```