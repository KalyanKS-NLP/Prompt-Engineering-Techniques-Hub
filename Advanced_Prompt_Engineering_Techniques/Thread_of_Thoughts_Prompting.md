# **Thread of Thoughts Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Thread-of-Thoughts (ThoT) prompting is a technique designed to help LLMs handle chaotic or cluttered contexts — especially when the input contains many irrelevant passages, mixed information, or retrieved evidence scattered across locations. ThoT prompting is triggered using the phrase, *“Walk me through this context in manageable parts step by step, summarizing and analyzing as we go.”*

While Chain-of-Thought (CoT) helps reasoning by “thinking step by step,” ThoT prompting goes further:

- ThoT breaks long/chaotic context into manageable parts.

- Summarizes each part.

- Identifies the relevant pieces.

- Then synthesizes the final answer.

It is extremely useful in *retrieval-augmented generation*, *multi-turn dialogue*, or *any situation* where a lot of irrelevant text is mixed with relevant information.

![Thread of Thoughts prompting](images/5-tot-prompt.jpg)

Figure from [Thread of Thoughts prompting](https://arxiv.org/abs/2311.08734) paper. 

## **Prompt Template**

Here is the prompt template for thread of thoughts prompting.

```
You are an assistant that performs Thread-of-Thoughts reasoning:

Context: 
{retrieved_passages}

Question: {question}

Trigger for Thread-of-Thoughts:
Walk me through this context in manageable parts step by step, summarizing and analyzing as we go.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of thread of thoughts promtping technique using LangChain v1.0

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

# 2. Define a Pydantic schema for structured ThoT output
class ThoTResponse(BaseModel):
    thread_of_thought: str = Field(..., description="Segment-by-segment analysis with summaries")
    answer: str = Field(..., description="Final answer extracted after analysis")

# 3. Create parser for the structured output
parser = PydanticOutputParser(pydantic_object=ThoTResponse)

# 4. Initialize the LLM (gemini-2.5-flash)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Thread-of-Thoughts prompt template (using your example)
prompt_template = ChatPromptTemplate.from_template(
    """
You are an assistant that performs Thread-of-Thoughts reasoning:

Context: 
{retrieved_passages}

Question: {question}

Trigger for Thread-of-Thoughts:
Walk me through this context in manageable parts step by step, summarizing and analyzing as we go.

Provide the output using this JSON format:
{format_instructions}
"""
)

# 6. Inject format instructions into prompt
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. LCEL chain: prompt → model → parser
chain = prompt | model | parser

# 8. Example data (your provided retrieval example)
retrieved_passages = """
Passage 1: Talks about book vending machines.
Passage 2: Reclam's founder created the publishing house in Leipzig.
Passage 3: Mentions a random street address.
Passage 4: Reclam's publishing house was located in Leipzig.
Passage 5: Talks about another unrelated company.
"""

question = "Where was Reclam founded?"

# 9. Invoke the chain
result = chain.invoke({
    "retrieved_passages": retrieved_passages,
    "question": question
})

# 10. Display the result
print("\n--- Thread of Thoughts ---\n", result.thread_of_thought)
print("\n--- Final Answer ---\n", result.answer)
```
Here the output is
```
--- Thread of Thoughts ---
 Let's break down the provided passages to find the answer to where Reclam was founded.

*   **Passage 1 Summary:** This passage discusses book vending machines.
*   **Passage 1 Analysis:** This passage does not contain any information relevant to Reclam's founding location.

*   **Passage 2 Summary:** This passage states that Reclam's founder created the publishing house in Leipzig.
*   **Passage 2 Analysis:** This passage directly answers the question, indicating that Reclam was founded in Leipzig.

*   **Passage 3 Summary:** This passage mentions a random street address.
*   **Passage 3 Analysis:** This passage is irrelevant to the question about Reclam's founding location.

*   **Passage 4 Summary:** This passage states that Reclam's publishing house was located in Leipzig.
*   **Passage 4 Analysis:** This passage corroborates the information from Passage 2, confirming Leipzig as the location of Reclam's publishing house.

*   **Passage 5 Summary:** This passage talks about another unrelated company.
*   **Passage 5 Analysis:** This passage is irrelevant to the question.

**Conclusion:** Based on Passage 2, which explicitly states that Reclam's founder created the publishing house in Leipzig, and Passage 4, which confirms its location in Leipzig, the founding location is clearly identified.

--- Final Answer ---
 Leipzig
```
