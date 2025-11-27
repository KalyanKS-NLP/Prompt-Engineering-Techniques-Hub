# **Zero Shot Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Zero-shot prompting is the simplest prompting technique where a large language model is given only an instruction or question (no examples) and is expected to complete the task using its general pre-trained knowledge.  It is like asking, *“Translate this sentence into French”* or *“Classify this review as positive, negative, or neutral”* without showing any sample translations or labeled reviews. In simple words, zero-shot prompting is prompting with a clear instruction or question without any examples. 

Zero-shot prompting is like a teacher giving a student a problem to solve without showing them a practice example on the board first. The student must rely solely on their general knowledge and what they have learned previously to arrive at the answer.

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation (News headlines classification)**

Here is the implementation of zero-shot prompting for news headlines classification.

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

Here is the implementation of zero-shot prompting for key phrases extraction.

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

# 5. Zero-shot prompt template for key phrase extraction
prompt_template = ChatPromptTemplate.from_template(
    """
Extract the most important key phrases from the text. 
Key phrases should be meaningful, concise, and capture the core concepts.

Text:
{input_text}

Provide the output in this JSON format:
{format_instructions}
"""
)

# 6. Inject parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build LCEL chain
chain = prompt | model | parser

# 8. Example text
input_text = "Artificial intelligence is transforming healthcare by enabling faster diagnosis, personalized treatments, and advanced medical imaging."

# 9. Invoke
result = chain.invoke({"input_text": input_text})

# 10. Display results
print("\n--- Key Phrases ---\n", result.key_phrases)
print("\n--- Reason ---\n", result.short_reason)
```

Here the output is
```
--- Key Phrases ---
 ['Artificial intelligence', 'transforming healthcare', 'faster diagnosis', 'personalized treatments', 'advanced medical imaging']
```


