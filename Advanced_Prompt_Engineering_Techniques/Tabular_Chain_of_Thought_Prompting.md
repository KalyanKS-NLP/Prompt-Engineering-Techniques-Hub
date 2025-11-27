# **Tabular Chain of Thought Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Tabular Prompting (Tab-CoT) is a prompting technique where the model is guided to show its reasoning in the form of a *table*, instead of plain step-by-step text.

Just like Zero-Shot CoT makes the model “think step by step,” Zero-Shot Tab-CoT makes the model think in a structured table format with clear columns such as:

| step | subquestion | process | result |

This table format forces the model to reason in a highly organized and structured way, which often leads to:

- More accurate reasoning
- Less confusion in multi-step problems
- Clearer intermediate results
- Better handling of numbers and logic

![Tabular Chain of Thought prompting](images/6-tcot-prompt.jpg)

Figure from [Tabular Chain of Thought prompting](https://arxiv.org/abs/2305.17812) paper. 

## **Prompt Template**

Here is the prompt template for tabular chain of thoughts prompting.

```
You are a reasoning assistant that uses Tabular Chain-of-Thought (Tab-CoT).

You must generate your reasoning in a table format using the header:

|step|subquestion|process|result|

For every step:
- Fill each column
- Show clean calculations in the "process" column
- Show only the intermediate numeric answer in "result"

After generating the full reasoning table, provide the final answer.

Question: {question}
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of tabular chain of thoughts promtping technique using LangChain v1.0

```python
# !pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --------------------------------------------------------
# 1. Set your API Key
# --------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# --------------------------------------------------------
# 2. Define the Pydantic schema for structured output
# --------------------------------------------------------
class TabCoTResponse(BaseModel):
    reasoning_table: str = Field(..., description="Generated Tabular Chain-of-Thought reasoning table")
    answer: str = Field(..., description="Final numeric answer only")

# --------------------------------------------------------
# 3. Create the parser
# --------------------------------------------------------
parser = PydanticOutputParser(pydantic_object=TabCoTResponse)

# --------------------------------------------------------
# 4. Initialize the model (Gemini-2.5-flash)
# --------------------------------------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# --------------------------------------------------------
# 5. Prompt Template for Zero-Shot Tabular CoT
# --------------------------------------------------------
prompt_template = ChatPromptTemplate.from_template(
    """
You are a reasoning assistant that uses **Tabular Chain-of-Thought (Tab-CoT)**.

You must generate your reasoning in a table format using the header:

|step|subquestion|process|result|

For every step:
- Fill each column
- Show clean calculations in the "process" column
- Show only the intermediate numeric answer in "result"

After generating the full reasoning table, provide the final answer.

Question: {question}

Provide the output in the following JSON format:
{format_instructions}
"""
)

# Insert parser format instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# --------------------------------------------------------
# 6. Build the LCEL chain
# --------------------------------------------------------
chain = prompt | model | parser

# --------------------------------------------------------
# 7. Example problem (YOUR GIVEN EXAMPLE)
# --------------------------------------------------------
question = (
    "A librarian is shelving books. A shelf for fiction novels can hold 15 books, "
    "and a shelf for non-fiction can hold 12 books. If the library needs to shelve "
    "90 fiction novels and 72 non-fiction books, how many total shelves will the librarian need?"
)

# --------------------------------------------------------
# 8. Invoke the chain
# --------------------------------------------------------
result = chain.invoke({"question": question})

# --------------------------------------------------------
# 9. Display results
# --------------------------------------------------------
print("\n--- Tabular Reasoning Table ---\n")
print(result.reasoning_table)

print("\n--- Final Answer ---\n")
print(result.answer)
```

Here the output is
```
--- Tabular Reasoning Table ---

|step|subquestion|process|result|
|---|---|---|---|
|1|How many shelves are needed for fiction novels?|90 novels / 15 novels/shelf|6|
|2|How many shelves are needed for non-fiction books?|72 books / 12 books/shelf|6|
|3|What is the total number of shelves needed?|6 (fiction shelves) + 6 (non-fiction shelves)|12|

--- Final Answer ---

12
```

