# **Few-Shot Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Few-shot prompting is a technique where you give a large language model a small number of example input-output pairs along with your instruction or question. In other words you show the model how a few instances of the task should be done, then ask it to apply the same pattern to a new instance.  In simple words, few-shot prompting = prompting with a clear instruction *and* a few example input-output pairs.

Few-shot prompting is like a teacher first shows a student a couple of solved problems on the board — “this is how you do it” — and then gives a new problem for the student to solve on their own. The student uses the pattern from the examples to work out the new problem.


## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation (News headlines classification)**

Here is the implementation of few-shot prompting for news headlines classification.

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
class FewShotClassifyResponse(BaseModel):
    predicted_label: str = Field(..., description="Predicted news category")

# 3. Create the parser
parser = PydanticOutputParser(pydantic_object=FewShotClassifyResponse)

# 4. Initialize the chat model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Few-shot prompt template (includes examples)
prompt_template = ChatPromptTemplate.from_template(
    """
Classify the news headline into one of:
["Politics", "Sports", "Business", "Technology", "Entertainment", "Health", "World"]

Here are some examples:

Example 1:
Headline: "Prime Minister meets foreign delegates to discuss trade agreements."
Label: Politics

Example 2:
Headline: "Tech company introduces new AI-powered smartphone."
Label: Technology

Example 3:
Headline: "Stock markets fall amid global economic slowdown."
Label: Business

Now classify the following:

Headline: {headline}

Provide your output in this JSON format:
{format_instructions}
"""
)

# 6. Inject parser formatting instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build the chain
chain = prompt | model | parser

# 8. Example headline
headline = "Government approves new policy to boost semiconductor manufacturing."

# 9. Invoke the chain
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

Here is the implementation of few-shot prompting for key phrases extraction.

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

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=KeyPhraseResponse)

# 4. Initialize model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Few-shot prompt with examples
prompt_template = ChatPromptTemplate.from_template(
    """
Extract the most important key phrases from the text. 
Key phrases should be meaningful, concise, and capture core concepts.

Here are some examples:

Example 1:
Text: "Climate change is accelerating due to rising greenhouse gas emissions."
Key Phrases: ["climate change", "greenhouse gas emissions"]

Example 2:
Text: "Machine learning models require large datasets for effective training."
Key Phrases: ["machine learning models", "large datasets", "effective training"]

Example 3:
Text: "Renewable energy sources like solar and wind are becoming more affordable."
Key Phrases: ["renewable energy sources", "solar", "wind", "affordable energy"]

Now extract key phrases from the following text:

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
```
Here the output is
```
--- Key Phrases ---
 ['artificial intelligence', 'healthcare transformation', 'faster diagnosis', 'personalized treatments', 'advanced medical imaging']
```