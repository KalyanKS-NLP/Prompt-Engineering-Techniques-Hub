# **Meta Cognitive Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Meta-Cognitive Prompting (MP) is a prompting technique that guides a Large Language Model (LLM) through a structured *self-reflection process*, mirroring how humans think about their own thinking.

Just as humans evaluate their initial interpretations, question their assumptions, and refine their understanding, MP forces the model to:

1. Understand the input
2. Make an initial judgment
3. Reflect on and critique this judgment
4. Form a final decision with justification
5. Assess its own confidence

![Meta Cognitive prompting](images/5-meta-cognitive-prompt.jpg)

Figure from [Meta Cognitive prompting](https://arxiv.org/abs/2308.05342) paper. 

## **Prompt Template**

Here is the prompt template for meta cognitive prompting.

```
For the question: "{question}" and statement: "{sentence}", determine if the statement provides the answer
to the question. If the statement contains the answer to the question, the status is entailment.
If it does not, the status is not_entailment. As you perform this task, follow these steps:

1. Clarify your understanding of the question and the context sentence.
2. Make a preliminary identification of whether the context sentence contains the answer to the question.
3. Critically assess your preliminary analysis. If you feel unsure about your initialentailment classification, try to reassess it.
4. Confirm your final answer and explain the reasoning behind your choice.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.

Provide the answer in your final response as "The status is (entailment / not_entailment)"

As you perform the above, produce the following structured output.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Zero-Shot Implementation**

Now let's see the implementation of few-shot meta cognitive promtping technique using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# 2. Define structured output for Meta-Cognitive Prompting (added final_answer field)
class MetaCognitiveResponse(BaseModel):
    understanding: str = Field(..., description="Clarify understanding of the question and the context sentence")
    preliminary_judgment: str = Field(..., description="Initial assessment of whether the statement contains the answer")
    critical_evaluation: str = Field(..., description="Reflection and reassessment of the initial judgment")
    final_answer: str = Field(..., description='Final response in the exact form: "The status is (entailment / not_entailment)"')
    confidence: str = Field(..., description="Confidence score (0-100%) with explanation")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=MetaCognitiveResponse)

# 4. Initialize Gemini model (gemini-2.5-flash)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Zero-Shot Meta-Cognitive Prompt Template (exact wording requested)
prompt_template = ChatPromptTemplate.from_template(
    """
For the question: "{question}" and statement: "{sentence}", determine if the statement provides the answer
to the question. If the statement contains the answer to the question, the status is entailment.
If it does not, the status is not_entailment. As you perform this task, follow these steps:
1. Clarify your understanding of the question and the context sentence.
2. Make a preliminary identification of whether the context sentence contains the answer to the question.
3. Critically assess your preliminary analysis. If you feel unsure about your initialentailment classification, try to reassess it.
4. Confirm your final answer and explain the reasoning behind your choice.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.
Provide the answer in your final response as "The status is (entailment / not_entailment)"

As you perform the above, produce the following structured output.

Provide your response in JSON format exactly matching the fields:
{format_instructions}
"""
)

# 6. Insert parser instructions
prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())

# 7. Build LCEL chain
chain = prompt | model | parser

# 8. Invoke the chain with your example question + statement
question = "What is the largest planet in our solar system?"
statement = (
    "Jupiter, the fifth planet from the Sun, is so massive that it accounts for more "
    "than twice the mass of all the other planets combined."
)

result = chain.invoke({
    "question": question,
    "sentence": statement
})

# 9. Output (structured)
print("\n--- Understanding ---\n", result.understanding)
print("\n--- Preliminary Judgment ---\n", result.preliminary_judgment)
print("\n--- Critical Evaluation ---\n", result.critical_evaluation)
print("\n--- Final Answer ---\n", result.final_answer)
print("\n--- Confidence ---\n", result.confidence)
```

Here the output is
```
--- Understanding ---
 The question asks for the name of the planet that is the largest in our solar system. The term 'largest' can refer to size (diameter/volume) or mass. The statement provides information about Jupiter's mass relative to all other planets.

--- Preliminary Judgment ---
 The statement identifies Jupiter and describes it as 'so massive that it accounts for more than twice the mass of all the other planets combined.' This strongly implies that Jupiter is the largest planet, at least in terms of mass. Given that 'largest' often encompasses mass when referring to celestial bodies, the statement appears to provide the answer.

--- Critical Evaluation ---
 The question asks 'What is the largest planet?'. The statement names 'Jupiter' and describes its extreme 'massiveness' ('so massive that it accounts for more than twice the mass of all the other planets combined'). While 'largest' can strictly mean largest in diameter or volume, being 'so massive' to that extent makes Jupiter unequivocally the largest by mass. In common astronomical discourse, the most massive planet is also considered the 'largest' in a general sense. The statement directly provides the name of the planet and a superlative characteristic (its mass) that confirms its status as the largest. Therefore, the statement does provide the answer to the question.

--- Final Answer ---
 The status is entailment

--- Confidence ---
 100%. The statement explicitly names Jupiter and provides a superlative description of its mass, which directly answers the question of which planet is the 'largest' in our solar system, as mass is a primary measure of a planet's 'largeness'.
```


## **Few-Shot Implementation**

Now let's see the implementation of few-shot meta cognitive promtping technique using LangChain v1.0

```python
# !pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# 2. Define Pydantic schema for Meta-Cognitive output
class MetaCognitiveResponse(BaseModel):
    understanding: str = Field(..., description="Clarify understanding of the question and the context sentence")
    preliminary_judgment: str = Field(..., description="Initial assessment of whether the statement contains the answer")
    critical_evaluation: str = Field(..., description="Reflection and reassessment of the initial judgment")
    final_answer: str = Field(..., description='Final response in the form: "The status is (entailment / not_entailment)"')
    confidence: str = Field(..., description="Confidence score (0-100%) with explanation")

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=MetaCognitiveResponse)

# 4. Initialize Gemini model (few-shot)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)

# 5. Few-shot Meta-Cognitive example
few_shot_example = """
Goal: Solve the problem using **Meta-Cognitive Prompting**.

Problem:
Question (Q): What is the largest planet in our solar system?
Statement (S): Jupiter, the fifth planet from the Sun, is so massive that it accounts for more
than twice the mass of all the other planets combined.

--- Understanding ---
The question asks for the name of the planet that is the largest in our solar system.
The statement provides information about Jupiter and describes its extreme mass.

--- Preliminary Judgment ---
The statement identifies Jupiter and describes it as exceptionally massive, strongly
suggesting it is the largest planet. The statement appears to contain the answer.

--- Critical Evaluation ---
The question asks for the "largest planet." The statement names Jupiter and describes
its mass as exceeding that of all other planets combined. Although "largest" can refer
to diameter or volume, being overwhelmingly massive supports its classification as the
largest. The statement directly provides the planet's name and evidence for its status.
Thus, the statement does provide the answer.

--- Final Answer ---
The status is entailment

--- Confidence ---
100%. The statement explicitly names Jupiter and provides a superlative description
of its mass, directly answering the question about the largest planet.
"""

# 6. Few-shot Meta-Cognitive Prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert reasoning assistant.

Below is an example problem solved using **Meta-Cognitive Prompting**:
{few_shot_example}

Now apply the same Meta-Cognitive structure to solve the following problem:

Question (Q): {question}
Statement (S): {statement}

As you perform this task, follow these steps exactly:

1. Clarify your understanding of the question and the context sentence.
2. Make a preliminary identification of whether the context sentence contains the answer.
3. Critically assess your preliminary analysis. If you feel unsure about your initial
   entailment classification, try to reassess it.
4. Confirm your final answer and explain the reasoning behind your choice.
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation
   for this confidence level.

Provide the answer in your final response as:
"The status is (entailment / not_entailment)"

Finally, produce your complete response in the following JSON format:
{format_instructions}
"""
)

# 7. Insert few-shot example and parser instructions
prompt = prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

# 8. Build LCEL chain
chain = prompt | model | parser

# 9. Target problem
question = "Which explorer was the first to circumnavigate the globe?"
statement = (
    "Ferdinand Magellan initiated the first sea voyage to sail all the way around the "
    "world, although he was killed in the Philippines before the journey was completed."
)

# 10. Run the chain
result = chain.invoke({
    "question": question,
    "statement": statement
})

# 11. Display structured result
print("\n--- Understanding ---\n", result.understanding)
print("\n--- Preliminary Judgment ---\n", result.preliminary_judgment)
print("\n--- Critical Evaluation ---\n", result.critical_evaluation)
print("\n--- Final Answer ---\n", result.final_answer)
print("\n--- Confidence ---\n", result.confidence)
```

Here the output is
```
--- Understanding ---
 The question asks for the name of the individual explorer who was the first to successfully complete a journey around the entire globe. The statement provides information about Ferdinand Magellan, stating that he initiated the first sea voyage that aimed to sail all the way around the world, but explicitly notes that he died before completing the journey himself.

--- Preliminary Judgment ---
 The statement names Ferdinand Magellan and describes his role in the first circumnavigation voyage. However, it also clearly states that he did not complete the journey. This suggests that while he was instrumental, he might not be the answer to 'who was the first to circumnavigate'. Therefore, the statement likely does not contain the answer to the question as phrased.

--- Critical Evaluation ---
 The question specifically asks 'Which explorer was the first to *circumnavigate* the globe?' To 'circumnavigate' means to travel all the way around. The statement explicitly says that Ferdinand Magellan 'was killed in the Philippines before the journey was completed.' This means Magellan himself did not complete the circumnavigation. While his expedition was the first to do so, he personally was not the first explorer to achieve it. Therefore, the statement does not provide the answer to the question; in fact, it provides information that disqualifies Magellan as the answer to the question as phrased. The question is about the individual's achievement, not just the initiation of the voyage.

--- Final Answer ---
 The status is not_entailment

--- Confidence ---
 100%. The statement directly contradicts Magellan being the first to *complete* the circumnavigation by explicitly stating he died before the journey was completed. The question asks for the explorer who *was* the first to circumnavigate, implying completion by that individual.
 ```