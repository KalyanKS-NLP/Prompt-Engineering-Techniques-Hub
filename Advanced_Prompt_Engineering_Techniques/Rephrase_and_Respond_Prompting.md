# **Rephrase and Respond Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Rephrase-and-Respond Prompting is a technique in which the model improves the clarity of a user’s question by first rewriting (rephrasing) the question in a clearer, more explicit form, and then answering that clarified version.  

The model:

1. Rephrases the question to make all implicit information explicit
2. Expands the intent, adds needed details, and clarifies categories or constraints
3. Responds to the improved version of the question

By strengthening the question before solving it, the model reduces errors caused by misinterpretation, vague phrasing, or missing details.

![Rephrase and Respond prompting](images/1-rephrase-respond-prompt.jpg)

Figure from [Rephrase and Respond prompting](https://arxiv.org/abs/2311.04205) paper.

##  **Prompt Template**

Here is the prompt template for rephrase and respond prompting.

```
You are an expert reasoning assistant.

For the user question below, perform BOTH steps in a single reasoning flow:

1. Rephrase and expand the question  
   - Remove ambiguity  
   - State the hidden intention clearly  
   - Make the required reasoning explicit  

2. Respond to the rephrased question  
   - Follow the clarified interpretation  
   - Provide a correct and well-reasoned answer  

User Question:
{question}
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of rephrase and respond promtping technique using LangChain v1.0

```python
# !pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# ----------------------------------------------------------
# 1. Set Gemini API Key
# ----------------------------------------------------------

os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ----------------------------------------------------------
# 2. Structured Output for Rephrase-and-Respond
# ----------------------------------------------------------

class RaRResult(BaseModel):
    rephrased_question: str = Field(..., description="The rephrased and expanded question")
    response: str = Field(..., description="Final answer produced after rephrasing")


rar_parser = PydanticOutputParser(pydantic_object=RaRResult)


# ----------------------------------------------------------
# 3. Initialize Gemini model
# ----------------------------------------------------------

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)


# ----------------------------------------------------------
# 4. Single-Prompt Rephrase-and-Respond Template
# ----------------------------------------------------------

rar_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert reasoning assistant.

For the user question below, perform BOTH steps in a single reasoning flow:

1. Rephrase and expand the question  
   - Remove ambiguity  
   - State the hidden intention clearly  
   - Make the required reasoning explicit  

2. Respond to the rephrased question  
   - Follow the clarified interpretation  
   - Provide a correct and well-reasoned answer  

User Question:
{question}

Provide your output in this JSON format:
{format_instructions}
"""
)

rar_prompt = rar_prompt_template.partial(
    format_instructions=rar_parser.get_format_instructions()
)


# ----------------------------------------------------------
# 5. Build the LCEL Chain — Only One LLM Call
# ----------------------------------------------------------

rar_chain = rar_prompt | model | rar_parser


# ----------------------------------------------------------
# 6. Run RaR on the Example Question
# ----------------------------------------------------------

question = "Identify the odd one out: Apple, Banana, Car, Orange."

result = rar_chain.invoke({"question": question})

print("\n--- REPHRASED QUESTION ---\n")
print(result.rephrased_question)

print("\n--- FINAL RESPONSE ---\n")
print(result.response)
```

Here the output is
```
--- REPHRASED QUESTION ---

Given the list of items 'Apple', 'Banana', 'Car', and 'Orange', identify which item is the 'odd one out' by determining a common semantic category that applies to three of the items, and then explicitly stating why the remaining item does not fit into that established category. The reasoning for the categorization must be clear.

--- FINAL RESPONSE ---

The odd one out is 'Car'.

**Reasoning:**
*   'Apple' is a type of fruit.
*   'Banana' is a type of fruit.
*   'Orange' is a type of fruit.
*   'Car' is a type of vehicle or mode of transportation.

Therefore, 'Apple', 'Banana', and 'Orange' all belong to the category of 'fruits', while 'Car' belongs to a completely different category, making it the odd one out.
```



