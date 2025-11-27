# **Analogical Promptiing**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**
Analogical prompting is a reasoning technique where the LLM is instructed to recall similar problems (analogies) before solving the main problem. Instead of giving the model examples yourself (as in few-shot prompting), you tell the model to generate its own relevant examples, just like a human remembering past problems to guide their thinking. 

By recalling similar problems first, the model creates context, activates the right concepts, and then solves the actual problem more accurately. 

![Analogical prompting](images/4-analogical-prompt.jpg)

Figure from [Analogical prompting ](https://arxiv.org/abs/2310.01714) paper. 

## **Prompt Template**
Here is the prompt template for analogical prompting.

```
Your task is to tackle mathematical problems. When presented with a math problem, recall relevant problems as examples. Afterward, proceed to solve the initial problem.

# Problem:
{question}

# Instructions:
## Relevant Problems:
Recall three examples of math problems that are relevant to the initial problem. Your problems should be distinct from each other and from the initial problem (e.g., involving different numbers and names). For each problem:
- After "Q: ", describe the problem
- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.

## Solve the Initial Problem:
Q: Copy and paste the initial problem here.
A: Explain the solution step by step and enclose the final answer in \\boxed{{}}.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of analogical promtping technique using LangChain v1.0

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

# 2. Define the structured output schema
class AnalogicalResponse(BaseModel):
    relevant_problems: str = Field(..., description="Self-generated relevant example problems with solutions")
    reasoning_chain: str = Field(..., description="Step-by-step reasoning for the original problem")
    answer: str = Field(..., description="Final numeric answer only")

# 3. Create the parser
parser = PydanticOutputParser(pydantic_object=AnalogicalResponse)

# 4. Initialize the chat model (Gemini 2.5 Flash)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Analogical prompting template (matches the structure in the image)
prompt_template = ChatPromptTemplate.from_template(
    """
Your task is to tackle mathematical problems. When presented with a math problem, recall relevant problems as examples. Afterward, proceed to solve the initial problem.

# Problem:
{question}

# Instructions:
## Relevant Problems:
Recall three examples of math problems that are relevant to the initial problem. Your problems should be distinct from each other and from the initial problem (e.g., involving different numbers and names). For each problem:
- After "Q: ", describe the problem
- After "A: ", explain the solution and enclose the ultimate answer in \\boxed{{}}.

## Solve the Initial Problem:
Q: Copy and paste the initial problem here.
A: Explain the solution step by step and enclose the final answer in \\boxed{{}}.

Provide the final output in the following JSON format:
{format_instructions}
"""
)

# 6. Inject format instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build LCEL chain (prompt → model → parser)
chain = prompt | model | parser

# 8. Example problem (your chosen analogical example)
question = "What is the area of the rectangle with the four vertices at (1, 3), (7, 3), (7, 8), and (1, 8)?"

# 9. Invoke the chain
result = chain.invoke({"question": question})

# 10. Display results
print("\n--- Relevant Problems (Self-Generated) ---\n", result.relevant_problems)
print("\n--- Reasoning Chain ---\n", result.reasoning_chain)
print("\n--- Final Answer ---\n", result.answer)
```
Here the output is

```
--- Relevant Problems (Self-Generated) ---
 Q: A rectangular garden is 10 meters long and 7 meters wide. What is its area?
A: The area of a rectangle is calculated by multiplying its length by its width.
Area = Length × Width
Area = 10 meters × 7 meters
Area = 70 square meters.
The area of the garden is \boxed{70} square meters.

Q: What is the area of a rectangle with vertices at (2, 1), (8, 1), (8, 5), and (2, 5)?
A: To find the area, we need the length and width of the rectangle.
Let's take two adjacent vertices, for example, (2, 1) and (8, 1). The distance between these points gives one side length. Since the y-coordinates are the same, this is a horizontal side.
Length = |8 - 2| = 6 units.
Now, take another adjacent vertex, (8, 5), which shares an x-coordinate with (8, 1). The distance between (8, 1) and (8, 5) gives the other side length (width). Since the x-coordinates are the same, this is a vertical side.
Width = |5 - 1| = 4 units.
Area = Length × Width = 6 × 4 = 24 square units.
The area of the rectangle is \boxed{24} square units.

Q: A square has vertices at (-2, -1), (3, -1), (3, 4), and (-2, 4). What is its area?
A: To find the area of a square, we need the length of one of its sides.
Let's find the distance between two adjacent vertices, for example, (-2, -1) and (3, -1).
Side length = |3 - (-2)| = |3 + 2| = 5 units.
Since it's a square, all sides are equal.
Area = Side × Side = 5 × 5 = 25 square units.
The area of the square is \boxed{25} square units.

--- Reasoning Chain ---
 Q: What is the area of the rectangle with the four vertices at (1, 3), (7, 3), (7, 8), and (1, 8)?
A:
1.  **Identify the coordinates:** The given vertices are A=(1, 3), B=(7, 3), C=(7, 8), and D=(1, 8).
2.  **Determine the length of the sides:**
    *   We can find the length of the horizontal sides by looking at the change in x-coordinates when the y-coordinate is constant. For example, consider the side connecting (1, 3) and (7, 3). The length is the absolute difference of the x-coordinates: |7 - 1| = 6 units.
    *   We can find the length of the vertical sides by looking at the change in y-coordinates when the x-coordinate is constant. For example, consider the side connecting (7, 3) and (7, 8). The length is the absolute difference of the y-coordinates: |8 - 3| = 5 units.
3.  **Identify length and width:** From the calculations, one side of the rectangle has a length of 6 units, and the adjacent side has a length of 5 units. These represent the length and width of the rectangle.
4.  **Calculate the area:** The area of a rectangle is given by the formula Area = Length × Width.
    Area = 6 × 5 = 30 square units.
The area of the rectangle is \boxed{30}.

--- Final Answer ---
 30
```