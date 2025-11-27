# **Few-Shot Chain of Thought Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Few-shot Chain-of-Thought Prompting is a technique in which you provide the model with one or more solved examples, where each example contains:

- A question
- A step-by-step chain of thought
- The final answer

By showing the model *how* to think step by step through an example, you teach it the format and reasoning style that it should follow when answering a new but related question.

Unlike zero-shot CoT, where the model is only instructed to *“think step by step,”* few-shot CoT gives the model an actual demonstration of how to reason. When the model sees a worked-out example, it generalizes the reasoning pattern and applies it to new, unseen problems.

![Few-Shot Chain of Thought prompting](images/1-few-cot-prompt.jpg)

Figure from [Few-Shot Chain of Thought prompting](https://arxiv.org/abs/2201.11903) paper. 

## **Prompt Template**

Here is the prompt template for few-shot chain of thoughts prompting.

```
You are a step-by-step reasoning assistant.

Here is an example problem solved using chain-of-thought:
{few_shot_example}

Now solve the following question using a similar chain-of-thought approach:

Question: {question}

Answer:  
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of few-shot chain thoughts promtping technique using LangChain v1.0

```python

#!pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# 2. Define Pydantic schema
class CoTResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Step-by-step reasoning")
    answer: str = Field(..., description="Final numeric/date answer only")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=CoTResponse)

# 4. Initialize Gemini model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Few-shot CoT example (1-shot)
few_shot_example = """
Question: The historical event was originally planned for 11/05/1852, 
but due to unexpected weather, it was moved forward by two days to today. 
What is the date 8 days from today in MM/DD/YYYY?

Answer: Moving an event forward by two days from 11/05/1852 means today's date is 11/03/1852 (11/05/1852 - 2).
8 days from today (11/03/1852) is 11/11/1852.
So the answer is 11/11/1852.
"""

# 6. Few-shot prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
You are a step-by-step reasoning assistant.

Here is an example problem solved using chain-of-thought:
{few_shot_example}

Now solve the following question using a similar chain-of-thought approach:

Question: {question}

Answer:  

Provide your solution in the following JSON format:
{format_instructions}
"""
)

# 7. Inject reference example + parser formatting
prompt = prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

# 8. Build the LCEL chain
chain = prompt | model | parser

# 9. Target Question
question = (
    "A construction project started on 09/15/2024. The first phase took 12 days. "
    "The second phase was originally scheduled for 20 days, but was shortened by 3 days. "
    "What is the completion date of the second phase in MM/DD/YYYY?"
)

# 10. Run the chain
result = chain.invoke({"question": question})

# 11. Display result
print("\n--- Reasoning Chain ---\n", result.reasoning_chain)
print("\n--- Final Answer ---\n", result.answer)

```
Here the output is

```
--- Reasoning Chain ---
 The construction project started on 09/15/2024. The first phase took 12 days, so it ended on 09/15/2024 + 12 days = 09/27/2024. The second phase was originally scheduled for 20 days but was shortened by 3 days, meaning its actual duration was 20 - 3 = 17 days. To find the completion date of the second phase, we add 17 days to the end date of the first phase (09/27/2024). Adding 3 days to 09/27/2024 brings us to 09/30/2024 (since September has 30 days). We still need to add 17 - 3 = 14 more days. These 14 days will be in October. Therefore, the completion date of the second phase is 10/14/2024.

--- Final Answer ---
 10/14/2024
```
