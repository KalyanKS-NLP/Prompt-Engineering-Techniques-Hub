# **Zero Shot Chain of Thought Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**
Zero-shot Chain-of-Thought (CoT) prompting is a technique where you instruct an LLM to think step-by-step before generating the final answer.  

Here, “zero-shot” means the model gets no examples from you, and “chain-of-thought” means the model shows its reasoning steps before giving the final answer. 

![zero shot cot prompting](images/1-zs-cot-prompt.jpg)

Figure from [zero shot CoT prompting ](https://arxiv.org/abs/2205.11916) paper. 

## **Prompt Temtplate**

Here is the prompt template for zero shot CoT prompting.

```
You are a step-by-step reasoning assistant.

Question: {question}

Answer: Let's think step by step.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of zero shot CoT prompting using LangChain v1.0 library.

```python
!pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# 2. Define the Pydantic schema for structured output
class CoTResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Step-by-step reasoning")
    answer: str = Field(..., description="Final numeric answer only")

# 3. Create the parser from the Pydantic model
parser = PydanticOutputParser(pydantic_object=CoTResponse)

# 4. Initialize the chat model (gpt-4o-mini)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider = "google_genai",
    temperature=0
)

# 5. Prompt template with explicit zero-shot CoT cue ("Let's think step by step.")
prompt_template = ChatPromptTemplate.from_template(
    """
You are a step-by-step reasoning assistant.

Question: {question}

Answer: Let's think step by step.

Provide your solution in the following JSON format:
{format_instructions}

"""
)

# 6. Inject the parser's format instructions into the template
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build the LCEL chain (prompt → model → parser)
chain = prompt | model | parser

# 8. Example question and invocation
question = "A baker made 24 cookies. Half are chocolate chip. Half of those have sprinkles. How many chocolate-chip cookies with sprinkles?"

result = chain.invoke({"question": question})

# 9. Display the result
print("\n--- Reasoning Chain ---\n", result.reasoning_chain)
print("\n--- Final Answer ---\n", result.answer)

```
The output for the above code is

```
--- Reasoning Chain ---
1. The baker made a total of 24 cookies.
2. Half of these cookies are chocolate chip. So, we calculate 24 / 2 = 12 chocolate chip cookies.
3. Half of the chocolate chip cookies have sprinkles. So, we calculate 12 / 2 = 6 chocolate chip cookies with sprinkles.

--- Final Answer ---
 6
```