# **Self Ask Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Self-Ask prompting is an advanced prompting technique where the LLM learns how to break down a complex question into smaller follow-up questions by looking at one or more example demonstrations provided in the prompt.

In Self-Ask prompting, the model:

1. Checks whether follow-up questions are needed.
2. Asks itself a sub-question.
3. Answers it.
4. Asks another follow-up question.
5. Continues until it has enough information.
6. Outputs: “So the final answer is: …”

This structured decomposition improves reasoning, especially for multi-step or compositional problems.


![Self Ask prompting](images/2-self-ask-prompt.jpg)

Figure from [Self Ask prompting](https://arxiv.org/abs/2210.03350) paper. 


## **Prompt Template**

Here is the prompt template for few-shot chain of thoughts prompting.

```
Here is an example problem solved using self-ask prompting:
{few_shot_example}

Now solve the following question using a similar self-ask prompting approach:

Question: {question}
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of self ask promtping technique using LangChain v1.0

```python
!pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# 2. Define Pydantic schema
class SelfAskResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Complete self-ask transcript (follow-ups + intermediate answers)")
    answer: str = Field(..., description="Final answer only in MM/DD/YYYY format")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=SelfAskResponse)

# 4. Initialize Gemini model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Few-shot Self-Ask example (1-shot)
few_shot_example = """
Q: The historical event was originally planned for 11/05/1852, but due to unexpected weather, it was moved forward by two days to today. What is the date 8 days from today in MM/DD/YYYY?
Are follow up questions needed here: Yes.
Follow up: What is today's date?
Intermediate answer: Moving an event forward by two days from 11/05/1852 means today's date is 11/03/1852.
Follow up: What date is 8 days from today?
Intermediate answer: 8 days from 11/03/1852 is 11/11/1852.
So the final answer is: 11/11/1852.
"""

# 6. Prompt template matching your exact requested pattern
prompt_template = ChatPromptTemplate.from_template(
    """
You are a step-by-step reasoning assistant.

Here is an example problem solved using self-ask prompting:
{few_shot_example}

Now solve the following question using a similar self-ask prompting approach:

Question: {question}

Provide your solution in the following JSON format:
{format_instructions}
"""
)

# 7. Inject reference example + parser formatting into the prompt
prompt = prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

# 8. Build the LCEL chain
chain = prompt | model | parser

# 9. Target Question (given earlier)
question = (
    "A construction project started on 09/15/2024. The first phase took 12 days. "
    "The second phase was originally scheduled for 20 days, but was shortened by 3 days. "
    "What is the completion date of the second phase in MM/DD/YYYY?"
)

# 10. Run the chain
result = chain.invoke({"question": question})

# 11. Display result
print("\n--- Reasoning Chain (self-ask transcript) ---\n", result.reasoning_chain)
print("\n--- Final Answer ---\n", result.answer)


```
Here the output is
```
--- Reasoning Chain (self-ask transcript) ---
 Are follow up questions needed here: Yes.
Follow up: What is the completion date of the first phase?
Intermediate answer: The first phase started on 09/15/2024 and took 12 days. Adding 12 days to 09/15/2024 gives 09/27/2024. So, the first phase completed on 09/27/2024.
Follow up: What is the actual duration of the second phase?
Intermediate answer: The second phase was originally scheduled for 20 days but was shortened by 3 days. So, the actual duration is 20 - 3 = 17 days.
Follow up: What is the completion date of the second phase?
Intermediate answer: The second phase starts immediately after the first phase completes, which is 09/27/2024. It takes 17 days. Adding 17 days to 09/27/2024:
September has 30 days. From 09/27/2024, there are 3 days left in September (09/28, 09/29, 09/30).
17 days - 3 days = 14 days remaining.
These 14 days will be in October. So, the date is 10/14/2024.
So the final answer is: 10/14/2024.

--- Final Answer ---
 10/14/2024
```