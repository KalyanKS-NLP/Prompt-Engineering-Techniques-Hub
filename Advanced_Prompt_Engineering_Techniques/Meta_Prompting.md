# **Meta Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Zero-shot Meta Prompting is a technique where you provide the model with a structured, example-free template that tells it how to solve the given problem.  Unlike Zero-Shot Chain-of-Thought (CoT), which tells the model to *“think step by step”*, Meta Prompting gives the model a full structured blueprint for solving a task. 

This structured blueprint includes *how to begin*, *what steps to follow*, *how to format the reasoning* and *how to present the final answer*. This technique makes the model perform well because the structure itself guides its reasoning process.

![Meta prompting](images/3-meta-prompt.jpg)

Figure from [Meta prompting ](https://arxiv.org/abs/2311.11482) paper. 

## **Prompt Template**

Here is the prompt template for meta prompting.

```
You are a structured reasoning assistant that solves the given problem following the given solution structure.

Problem: {question}

Solution Structure:
  Step 1: Begin the response with: "Let's think step by step."
  Step 2: Identify the important components of the problem.
  Step 3: Break the solution process into clear, logical steps.
  Step 4: Present the final result in a LaTeX formatted box, like: \\boxed{{value}}

Final Answer: Provide only the final numeric answer.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of meta promtping technique using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# 2. Define the Pydantic schema for Meta Prompting structured output
class MetaPromptResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Structured reasoning following the meta-prompt steps")
    answer: str = Field(..., description="Final numeric answer only")

# 3. Create the parser
parser = PydanticOutputParser(pydantic_object=MetaPromptResponse)

# 4. Initialize the chat model (gemini-2.5-flash)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Zero-Shot Meta Prompt Template (structure-focused)
prompt_template = ChatPromptTemplate.from_template(
    """
You are a structured reasoning assistant that solves the given problem following the given solution structure.

Problem: {question}

Solution Structure:
  Step 1: Begin the response with: "Let's think step by step."
  Step 2: Identify the important components of the problem.
  Step 3: Break the solution process into clear, logical steps.
  Step 4: Present the final result in a LaTeX formatted box, like: \\boxed{{value}}

Final Answer: Provide only the final numeric answer.

Return your response using this JSON format:
{format_instructions}
"""
)

# 6. Inject parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build the LCEL chain
chain = prompt | model | parser

# 8. Example mathematical question
question = "Solve for x: 3x + 12 = 39."

# 9. Run the chain
result = chain.invoke({"question": question})

# 10. Display the structured reasoning and final answer
print("\n--- Reasoning Chain (Structured Meta Prompt) ---\n", result.reasoning_chain)
print("\n--- Final Answer ---\n", result.answer)
```

Here the output is
```
--- Reasoning Chain (Structured Meta Prompt) ---
 Let's think step by step.

Step 2: The important components of the problem are the linear equation 3x + 12 = 39 and the objective to solve for the variable x.

Step 3: We will solve the equation by isolating x through algebraic manipulation.

1.  Start with the given equation: 3x + 12 = 39
2.  To isolate the term containing x (3x), subtract 12 from both sides of the equation:
    3x + 12 - 12 = 39 - 12
    3x = 27
3.  To solve for x, divide both sides of the equation by 3:
    3x / 3 = 27 / 3
    x = 9

Step 4: The final result is \boxed{9}.

--- Final Answer ---
 9
```
