# **Batch Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Batch prompting is a prompting technique for large language models which involves giving the model multiple inputs (e.g., several questions or tasks) in a single prompt, instead of prompting once per input. The model generates all corresponding outputs in one go,  instead of generating output once per each input. It’s useful when you have many inputs to process (e.g., many reviews to classify, many sentences to translate, many questions to answer). By batching them, you reduce the number of separate inference calls needed, which cuts down token usage and inference time.

Batch prompting is like a teacher giving a student a whole worksheet with 10 questions at once (instead of one question at a time), the students solves all 10 questions at once.


## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation (News headlines classification)**

Here is the implementation of batch prompting for key phrases extraction.

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
class BatchClassifyResponse(BaseModel):
    predictions: List[str] = Field(..., description="Predicted labels for each headline")

# 3. Create the parser
parser = PydanticOutputParser(pydantic_object=BatchClassifyResponse)

# 4. Initialize the chat model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Batch prompting template
prompt_template = ChatPromptTemplate.from_template(
    """
You will classify multiple news headlines into one of the categories:
["Politics", "Sports", "Business", "Technology", "Entertainment", "Health", "World"]

Headlines:
{headlines}

Return the predictions in order, inside this JSON format:
{format_instructions}
"""
)

# 6. Inject parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Chain: prompt → model → parser
chain = prompt | model | parser

# 8. Example headlines (batch)
headlines = [
    "Government approves new policy to boost semiconductor manufacturing.",
    "Star striker leads team to victory in championship final.",
    "New study reveals long-term effects of poor sleep on health."
]

# 9. Invoke batch prediction
result = chain.invoke({"headlines": headlines})

# 10. Display results
print("\n--- Batch Predictions ---")
for i, label in enumerate(result.predictions):
    print(f"{i+1}. {label}")
```
Here the output is
```
--- Batch Predictions ---
1. Politics
2. Sports
3. Health
```


## **Implementation (Key phrases extraction)**

Here is the implementation of batch prompting for key phrases extraction.

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
class BatchKeyPhraseResponse(BaseModel):
    all_key_phrases: List[List[str]] = Field(
        ..., 
        description="List of key phrase lists (one list per text input)"
    )

# 3. Create the parser
parser = PydanticOutputParser(pydantic_object=BatchKeyPhraseResponse)

# 4. Initialize the chat model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Batch prompting template for key phrase extraction
prompt_template = ChatPromptTemplate.from_template(
    """
Extract the most important key phrases from each text below.
Key phrases must be meaningful, concise, and capture the core concepts.

Texts:
{texts}

Provide the output strictly in this JSON format:
{format_instructions}
"""
)

# 6. Inject parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build the chain
chain = prompt | model | parser

# 8. Example batch of texts
texts = [
    "Artificial intelligence is transforming healthcare by enabling faster diagnosis and advanced medical imaging.",
    "Climate change is accelerating due to rising greenhouse gas emissions and deforestation.",
    "Quantum computing promises exponential speedups for complex problem solving."
]

# 9. Invoke
result = chain.invoke({"texts": texts})

# 10. Display results
print("\n--- Batch Key Phrases ---")
for i, phrases in enumerate(result.all_key_phrases):
    print(f"\nText {i+1} key phrases:")
    print(phrases)
```

Here the output is
```
--- Batch Key Phrases ---

Text 1 key phrases:
['Artificial intelligence', 'healthcare transformation', 'faster diagnosis', 'advanced medical imaging']

Text 2 key phrases:
['Climate change acceleration', 'greenhouse gas emissions', 'deforestation']

Text 3 key phrases:
['Quantum computing', 'exponential speedups', 'complex problem solving']
```