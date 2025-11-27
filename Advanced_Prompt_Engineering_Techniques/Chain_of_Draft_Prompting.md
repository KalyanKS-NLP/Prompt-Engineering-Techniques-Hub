# **Chain of Draft Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Chain-of-Draft (CoD) prompting is a reasoning technique where you tell the LLM to think step-by-step but in a very short, compact form. In CoD prompting, the model still performs multi-step reasoning, but each step is extremely concise (often 3–5 words), focusing only on the essential information needed to progress. This is similar to how humans solve problems by scribbling tiny notes or equations instead of writing long explanations.

Chain of draft prompting reduces response length, token cost and response time while still keeping reasoning accuracy high. 

![chain of draft (CoD) prompting](images/2-cod-prompt.jpg)

Figure from [Chain of Draft prompting ](https://arxiv.org/abs/2502.18600) paper. 

## **Prompt Template**

Here is the prompt template for chain of draft (CoD) prompting.

```
You are a step-by-step reasoning assistant.

Question: {question}

Answer: Let's think step by step, but only keep a minimum draft for
each thinking step, with 5 words at most.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of chain of draft promtping technique using LangChain v1.0

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

Answer: Let's think step by step, but only keep a minimum draft for
each thinking step, with 5 words at most.

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

Here the output is
```
--- Reasoning Chain ---
 Total cookies: 24. Half are chocolate chip. 24 / 2 = 12. Half of those have sprinkles. 12 / 2 = 6. Final answer is 6.

--- Final Answer ---
 6
```
