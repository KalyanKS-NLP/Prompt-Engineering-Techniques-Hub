# **Chain of Translation Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Chain of Translation Prompting is a prompting technique where a non-English input sentence is first translated into English, and only then the actual task—such as sentiment analysis, emotion detection, toxicity classification, or hate-speech detection is performed.

This technique improves reliability because large language models are typically more accurate when processing English text. By inserting a translation step, the model gains access to richer semantic cues, clearer syntactic structure, and a more familiar linguistic environment. The result is more stable, consistent, and interpretable predictions compared to directly classifying the original sentence in a low-resource language.

![Chain of Translation prompting](images/1-chain-translation-prompt.jpg)

Figure from [Chain of Translation prompting](https://arxiv.org/abs/2409.04512) paper.

## **Prompt Template**

Here is the prompt template for chain of translation prompting.

```
Consider yourself to be a human annotator who is well versed in English and Telugu language.
Given a Telugu sentence as input, perform the following tasks on the sentence:

1. Translate the given Telugu sentence into English.

2. Identify the sentiment depicted by the sentence.
   If the sentence expresses a positive emotion or opinion, label it as Positive.
   If the sentence expresses a negative emotion or complaint, label it as Negative.
   If the sentence expresses neither positive nor negative sentiment, label it as Neutral.

3. Give the output as:
   - the original Telugu sentence,
   - its English translation,
   - and the sentiment label.

Sentence is as follows:
{sentence}
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of chain of translation promtping technique using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# ----------------------------------------------------------
# 1. Set your Gemini API key
# ----------------------------------------------------------

os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ----------------------------------------------------------
# 2. Define Final Structured Output Model
# ----------------------------------------------------------

class TranslationSentiment(BaseModel):
    telugu_sentence: str = Field(..., description="Original Telugu input")
    english_translation: str = Field(..., description="English translation of the Telugu text")
    sentiment_label: str = Field(..., description="Sentiment: Positive, Negative, or Neutral")


final_parser = PydanticOutputParser(pydantic_object=TranslationSentiment)


# ----------------------------------------------------------
# 3. Initialize Gemini model (single call)
# ----------------------------------------------------------

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)


# ----------------------------------------------------------
# 4. Single Prompt Template (Translation + Classification)
# ----------------------------------------------------------

prompt_template = ChatPromptTemplate.from_template(
    """
Consider yourself to be a human annotator who is well versed in English and Telugu language.
Given a Telugu sentence as input, perform the following tasks on the sentence:

1. Translate the given Telugu sentence into English.

2. Identify the sentiment depicted by the sentence.
   If the sentence expresses a positive emotion or opinion, label it as Positive.
   If the sentence expresses a negative emotion or complaint, label it as Negative.
   If the sentence expresses neither positive nor negative sentiment, label it as Neutral.

3. Give the output as:
   - the original Telugu sentence,
   - its English translation,
   - and the sentiment label.

Sentence is as follows:
{sentence}

Provide the final output using this JSON structure:
{format_instructions}
"""
)

single_prompt = prompt_template.partial(
    format_instructions=final_parser.get_format_instructions()
)


# ----------------------------------------------------------
# 5. Build LCEL Chain (single LLM call)
# ----------------------------------------------------------

chain = single_prompt | model | final_parser


# ----------------------------------------------------------
# 6. Run Chain of Translation Prompting on the Example
# ----------------------------------------------------------

telugu_sentence = "సినిమా అద్భుతంగా ఉంది! డైరెక్టర్ పనితీరు సూపర్. మళ్ళీ చూడాలని ఉంది."

result = chain.invoke({"sentence": telugu_sentence})

print("\n--- FINAL OUTPUT ---\n")
print("Telugu Sentence       :", result.telugu_sentence)
print("English Translation   :", result.english_translation)
print("Sentiment Label       :", result.sentiment_label)
```

Here the output is
```
--- FINAL OUTPUT ---

Telugu Sentence       : సినిమా అద్భుతంగా ఉంది! డైరెక్టర్ పనితీరు సూపర్. మళ్ళీ చూడాలని ఉంది.
English Translation   : The movie is wonderful! The director's work is super. I want to watch it again.
Sentiment Label       : Positive
```