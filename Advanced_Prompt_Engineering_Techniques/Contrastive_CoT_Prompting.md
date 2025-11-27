# **Contrastive Chain of Thought Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Contrastive Chain-of-Thought (Contrastive CoT) Prompting is an enhanced reasoning technique in which the model is given both a correct chain-of-thought and an incorrect chain-of-thought for a single example. 

This contrast teaches the model two things:

1. What good reasoning looks like (positive demonstration)
2. What kind of mistakes to avoid (negative demonstration)

This dual learning significantly improves reasoning performance, especially in tasks involving logic, arithmetic, dates, and multi-step reasoning. This is unlike regular few-shot CoT prompting, which gives only correct reasoning.

![Contrastive CoT prompting](images/3-contrastive-cot-prompt.jpg)

Figure from [Contrastive CoT prompting](https://arxiv.org/abs/2311.09277) paper. 


## **Prompt Template**

Here is the prompt template for few-shot chain of thoughts prompting.

```
You are a reasoning assistant that learns from contrastive examples.

Below is a contrastive demonstration containing:
- A correct chain-of-thought
- An incorrect chain-of-thought

Use the correct reasoning patterns and avoid mistakes shown in the wrong explanation.

Contrastive Demonstration:
{contrastive_example}

Now solve the following question using improved chain-of-thought reasoning:

Question: {question}
Answer: 
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of contrastive CoT promtping technique using LangChain v1.0

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
class ContrastiveCoTResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Step-by-step reasoning derived from correct signals")
    answer: str = Field(..., description="Final numeric/date answer only")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=ContrastiveCoTResponse)

# 4. Initialize Gemini model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Contrastive CoT example (1-shot)
contrastive_example = """
Q: The historical event was originally planned for 11/05/1852, 
but due to unexpected weather, it was moved forward by two days to today. 
What is the date 8 days from today in MM/DD/YYYY?

Correct Explanation:
Moving an event forward by two days from 11/05/1852 means today's date is 11/03/1852 (11/05/1852 - 2 days).
8 days from today (11/03/1852) is 11/11/1852.
So the answer is 11/11/1852.

Wrong Explanation:
Moving an event forward by two days from 11/05/1852 means today's date is 11/07/1852 (11/05/1852 + 2 days).
8 days from this incorrect 'today' (11/07/1852) would be 11/15/1852.
This is incorrect because "moved forward" means earlier in time, not later.
"""

# 6. Contrastive CoT prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
You are a reasoning assistant that learns from contrastive examples.

Below is a contrastive demonstration containing:
- A correct chain-of-thought
- An incorrect chain-of-thought

Use the correct reasoning patterns and avoid mistakes shown in the wrong explanation.

Contrastive Demonstration: 
{contrastive_example}

Now solve the following question using improved chain-of-thought reasoning:

Question: {question}
Answer: 

Provide your solution in the following JSON format:
{format_instructions}
"""
)

# 7. Inject example + parser instructions
prompt = prompt_template.partial(
    contrastive_example=contrastive_example,
    format_instructions=parser.get_format_instructions()
)

# 8. Build the LCEL chain
chain = prompt | model | parser

# 9. Target Question
question = (
    "The concert was scheduled to be on 06/01/1943, but was delayed by one day to today." 
    "What is the date 10 days ago in MM/DD/YYYY?"
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
 The concert was scheduled for 06/01/1943, but was delayed by one day to today. "Delayed by one day" means today's date is one day after the scheduled date. So, today's date is 06/01/1943 + 1 day = 06/02/1943.
We need to find the date 10 days ago from today (06/02/1943).
06/02/1943 - 10 days:
Subtracting 2 days from 06/02/1943 brings us to 05/31/1943 (since June 1st and 2nd are gone, and May has 31 days).
Remaining days to subtract: 10 - 2 = 8 days.
Subtracting 8 more days from 05/31/1943 gives us 05/23/1943.

--- Final Answer ---
 05/23/1943
```

