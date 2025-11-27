# **Chain of Symbol Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Chain-of-Symbol (CoS) Prompting is a reasoning technique in which the model is shown example solutions, but instead of using natural-language chain-of-thought, the examples use symbolic reasoning steps. In this each example contains:

- A question (natural language description of a spatial/stacking scenario)
- A symbolic representation of the intermediate steps (e.g., `U/T/R/S/V/W`)
- The final symbolic answer sequence

This technique is especially powerful in tasks involving, spatial relationships, Stack-order reasoning, object manipulation etc. 

![Chain of Symbol prompting](images/4-chain-symbol-prompt.jpg)

Figure from [Chain of Symbol prompting](https://arxiv.org/abs/2305.10276) paper. 

## **Prompt Template**

Here is the prompt template for chain of symbol prompting.

```
You are a symbolic spatial-reasoning assistant.

Here is an example problem solved using chain-of-symbol reasoning:
{few_shot_example}

Now solve the following question using a similar chain-of-symbol approach:

Question: {question}
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of chain of symbol promtping technique using LangChain v1.0

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

# 2. Define Pydantic schema
class CoSResponse(BaseModel):
    symbol_chain: str = Field(..., description="Symbolic step-by-step chain")
    answer: str = Field(..., description="Final sequence in symbolic format")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=CoSResponse)

# 4. Initialize Gemini model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Few-shot CoS example (1-shot)
few_shot_example = """
Q: There are a set of colored blocks. 
The blue block R is on top of the green block S. 
The red block T is on top of R. 
The yellow block U is on top of T. 
The green block S is on top of the orange block V. 
The orange block V is on top of the purple block W. 
We need to get block S. 
To grab a lower block, all blocks above it must be removed first. 
How to get block S?

A:
U/T/R/S/V/W
T/R/S/V/W
R/S/V/W
S/V/W
S
"""

# 6. Few-shot CoS prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
You are a symbolic spatial-reasoning assistant.

Here is an example problem solved using chain-of-symbol reasoning:
{few_shot_example}

Now solve the following question using a similar chain-of-symbol approach:

Question: {question}

Provide your solution in the following JSON format:
{format_instructions}
"""
)

# 7. Inject example + parser format instructions
prompt = prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

# 8. Build the LCEL chain
chain = prompt | model | parser

# 9. Target Question
question = (
    "There is a stack of four books. The Science book (S) is on the bottom. "
    "The History book (H) is on top of the Science book. "
    "The Math book (M) is on top of the History book. "
    "The Art book (A) is on the very top. "
    "We need to grab the Science book (S). To grab any book, all books above it must be removed first. "
    "What is the sequence of books to remove to get the Science book?"
)

# 10. Run the chain
result = chain.invoke({"question": question})

# 11. Display result
print("\n--- Symbol Chain ---\n", result.symbol_chain)
print("\n--- Final Answer ---\n", result.answer)
```
Here the output is

```
--- Symbol Chain ---
 A/M/H/S
M/H/S
H/S
S

--- Final Answer ---
 A, M, H, S
```