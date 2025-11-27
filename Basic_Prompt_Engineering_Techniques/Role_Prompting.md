# **Role Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Role prompting is a technique where the large language model is instructed to take on a specific *role, identity, or persona* before performing a task. Instead of giving only a task instruction, you tell the model who it should act as, such as *“Act as a cybersecurity expert and explain…”* or *“You are a professional journalist. Summarize…”*. By adopting the assigned role, the model adjusts its tone, depth, and reasoning to match that persona.

It is like asking a student to “pretend you are a doctor” before explaining a medical concept. The student now answers not just from general knowledge but through the lens of that specialized role. In simple words, role prompting guides the model’s behavior by assigning it a specific identity or expertise.

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation (News headlines classification)**

Here is the implementation of role prompting for news headlines classification.

```python
# !pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# 2. Define the Pydantic schema for structured output
class ZeroShotClassifyResponse(BaseModel):
    predicted_label: str = Field(..., description="Predicted news category")

# 3. Create the parser
parser = PydanticOutputParser(pydantic_object=ZeroShotClassifyResponse)

# 4. Initialize the chat model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Zero-shot prompt template (no examples)
prompt_template = ChatPromptTemplate.from_template(
    """
You are a professional news editor with years of experience in global journalism. Your job is to accurately classify news headlines into their correct category.
Classify the news headline into one of the categories:
["Politics", "Sports", "Business", "Technology", "Entertainment", "Health", "World"]

Headline: {headline}

Provide your output in this JSON format:
{format_instructions}
"""
)

# 6. Inject parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Chain: prompt → model → parser
chain = prompt | model | parser

# 8. Example headline
headline = "Government approves new policy to boost semiconductor manufacturing."

# 9. Invoke
result = chain.invoke({"headline": headline})

# 10. Display result
print("\n--- Predicted Label ---\n", result.predicted_label)
```

Here the output is
```
--- Predicted Label ---
 Politics
```


## **Implementation (Key phrases extraction)**

Here is the implementation of role prompting for key phrases extraction.

```python
# !pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# 2. Define the Pydantic schema for structured output
class KeyPhraseResponse(BaseModel):
    key_phrases: List[str] = Field(..., description="List of extracted key phrases")

# 3. Create the parser
parser = PydanticOutputParser(pydantic_object=KeyPhraseResponse)

# 4. Initialize the chat model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Role prompting template for key phrase extraction
prompt_template = ChatPromptTemplate.from_template(
    """
You are a professional linguistic analyst specializing in information extraction.
Your task is to extract the most important key phrases from the given text.

Key phrases should be:
- concise
- meaningful
- representative of the core ideas

Text:
{input_text}

Provide the output strictly in this JSON format:
{format_instructions}
"""
)

# 6. Inject parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build LCEL chain
chain = prompt | model | parser

# 8. Example text
input_text = (
    "Artificial intelligence is transforming healthcare by enabling faster diagnosis, "
    "personalized treatments, and advanced medical imaging."
)

# 9. Invoke
result = chain.invoke({"input_text": input_text})

# 10. Display results
print("\n--- Key Phrases ---\n", result.key_phrases)
```

Here the output is
```
--- Key Phrases ---
 ['Artificial intelligence', 'transforming healthcare', 'faster diagnosis', 'personalized treatments', 'advanced medical imaging']
```
