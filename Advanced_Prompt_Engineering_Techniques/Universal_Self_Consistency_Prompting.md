# **Universal Self Consistency Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Self-consistency prompting is a technique in which a large language model (LLM) is asked to generate multiple reasoning chains (via chain-of-thought prompting) for the same input, and then the final answer is chosen by majority vote (i.e., the answer that appears most frequently across the sampled outputs). The idea is that if many independent reasoning paths converge on the same answer, that answer is more likely to be correct.


Universal self-consistency prompting builds on self-consistency but removes its main limitation (that the final answer must be in a form that supports majority/exact-match voting). In USC, one again samples multiple outputs from the LLM, but then instead of simply voting on the same answer string, the LLM is prompted to select (or rank) among the candidate outputs which is the “most consistent” with the set of responses (or best according to some consistency criterion). This allows it to be applied to free-form generation tasks (summarization, open-ended Q&A, code generation) where answers are not identical strings and majority voting fails.


![Universal Self Consistency prompting](images/2-universal-self-prompt.jpg)

Figure from [Universal Self Consistency prompting](https://arxiv.org/abs/2311.17311) paper. 

## **Prompt Template**

Here is the generation prompt template for universal self consistency prompting.

```
You are a detailed step-by-step reasoning assistant.

Question: {question}

Instruction:
- Think step by step.
- Produce a clear chain of thought.
- Then produce ONLY the final numeric answer.
```
Here is the selection prompt template for universal self consistency prompting.

```
You are an evaluator assistant. You are given multiple candidate answers to the same question.
Your job is to read ALL responses and select the one that is the most consistent, reasonable, and logically sound.

Question:
{question}

Candidate Responses:
{all_responses}

Instruction:
- Carefully compare the reasoning steps.
- Select the single best response.
- Provide ONLY the index number of the best response.
- DO NOT explain your choice.

Return output in plain text containing ONLY the index number (1, 2, or 3)
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Zero-Shot Implementation**

Now let's see the implementation of zero-shot universal self consistency promtping technique using LangChain v1.0

```python
# ---------------------------------------------------------
# Zero-Shot Universal Self-Consistency Prompting (USC)
# ---------------------------------------------------------

# pip install langchain langchain-google-genai pydantic

import os
import time
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. Set your Gemini API key
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ---------------------------------------------------------
# 2. Define structured output model for candidate responses
# ---------------------------------------------------------
class USCResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Full reasoning steps")
    answer: str = Field(..., description="Final numeric answer only")


parser = PydanticOutputParser(pydantic_object=USCResponse)


# ---------------------------------------------------------
# 3. Initialize Gemini model with sampling enabled
# ---------------------------------------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.8,
    top_k=40,
)


# ---------------------------------------------------------
# 4. Zero-shot generation prompt (same as SC sampling stage)
# ---------------------------------------------------------
generation_prompt_template = ChatPromptTemplate.from_template(
    """
You are a detailed step-by-step reasoning assistant.

Question: {question}

Instruction:
- Think step by step.
- Produce a clear chain of thought.
- Then produce ONLY the final numeric answer.

Return output in this JSON format:
{format_instructions}
"""
)

generation_prompt = generation_prompt_template.partial(
    format_instructions=parser.get_format_instructions()
)

gen_chain = generation_prompt | model | parser


# ---------------------------------------------------------
# 5. Universal Self-Consistency Selection Prompt
# ---------------------------------------------------------
selection_prompt = ChatPromptTemplate.from_template(
    """
You are an evaluator assistant.

You are given multiple candidate answers to the same question.
Your job is to read ALL responses and select the one that is
the most consistent, reasonable, and logically sound.

Question:
{question}

Candidate Responses:
{all_responses}

Instruction:
- Carefully compare the reasoning steps.
- Select the single best response.
- Provide ONLY the index number of the best response.
- DO NOT explain your choice.

Return output in plain text containing ONLY the index number (1, 2, or 3).
"""
)

selection_chain = selection_prompt | model


# ---------------------------------------------------------
# 6. Universal Self-Consistency function
# ---------------------------------------------------------
def universal_self_consistency(question: str, n_samples: int = 3):
    candidates = []

    # --- Stage 1: Generate candidate responses ---
    for i in range(n_samples):
        result = gen_chain.invoke({"question": question})
        candidates.append(result)
        time.sleep(1)

    # Prepare text block for evaluation prompt
    formatted_candidates = ""
    for idx, c in enumerate(candidates, 1):
        formatted_candidates += (
            f"\n[{idx}] Reasoning:\n{c.reasoning_chain}\nAnswer: {c.answer}\n"
        )

    # --- Stage 2: Ask LLM to select best candidate ---
    chosen_idx = selection_chain.invoke(
        {
            "question": question,
            "all_responses": formatted_candidates,
        }
    )

    chosen_idx = int(chosen_idx.content.strip())

    return candidates[chosen_idx - 1], candidates


# ---------------------------------------------------------
# 7. Run Universal Self-Consistency on the example
# ---------------------------------------------------------
question = (
    "What are three advantages of electric vehicles over gasoline vehicles?"
)

best_output, all_candidates = universal_self_consistency(question, n_samples=3)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------
print("\n===== UNIVERSAL SELF CONSISTENCY OUTPUT =====")
print("Chosen Final Answer:", best_output.answer)

print("\n===== ALL GENERATED CANDIDATES =====")
for i, out in enumerate(all_candidates, 1):
    print(f"\n--- Candidate {i} ---")
    print(out.reasoning_chain)
    print("Answer:", out.answer)
```

Here the output is
```

===== UNIVERSAL SELF CONSISTENCY OUTPUT =====
Chosen Final Answer: 1. Zero tailpipe emissions, contributing to cleaner air and reduced greenhouse gases. 
2. Lower running costs due to cheaper 'fuel' (electricity) and significantly reduced maintenance requirements. 
3. Superior driving experience with instant torque for quick acceleration and quieter, smoother operation.

===== ALL GENERATED CANDIDATES =====

--- Candidate 1 ---
The user asked for three advantages of electric vehicles (EVs) over gasoline vehicles. I brainstormed several potential advantages, including environmental benefits, lower running costs, performance, and maintenance. From these, I selected three distinct and significant advantages:

1.  **Environmental Impact:** EVs produce zero tailpipe emissions, which significantly reduces local air pollution and greenhouse gas emissions compared to gasoline vehicles.
2.  **Lower Running Costs:** EVs generally have lower 'fuel' costs (electricity can be cheaper per mile than gasoline) and significantly reduced maintenance needs due to fewer moving parts (no oil changes, spark plugs, complex transmissions, etc.).
3.  **Performance and Driving Experience:** EVs offer instant torque, leading to quicker acceleration, and operate much more quietly and smoothly than gasoline cars, providing a superior driving experience.

Answer: 1. Zero tailpipe emissions, contributing to cleaner air and reduced greenhouse gases. 2. Lower running costs due to cheaper 'fuel' (electricity) and significantly reduced maintenance requirements. 3. Superior driving experience with instant torque for quick acceleration and quieter, smoother operation.

--- Candidate 2 ---
The user asked for three advantages of electric vehicles over gasoline vehicles. I will identify three distinct benefits:
1.  **Environmental Impact:** Electric vehicles produce zero tailpipe emissions, leading to cleaner air, especially in urban areas, and a reduction in smog-forming pollutants. While electricity generation might have emissions, the vehicle itself is clean.
2.  **Lower Running Costs:** Electricity is generally cheaper per mile than gasoline. Additionally, EVs have fewer moving parts than internal combustion engine vehicles, leading to lower maintenance requirements (no oil changes, spark plugs, fuel filters, etc.).
3.  **Performance and Driving Experience:** Electric motors provide instant torque, resulting in rapid and smooth acceleration. EVs are also significantly quieter than gasoline cars, offering a more serene driving experience.

Answer: 1. Zero tailpipe emissions
2. Lower running costs (cheaper fuel and less maintenance)
3. Instant torque and quieter operation

--- Candidate 3 ---
The user is asking for three advantages of electric vehicles (EVs) over gasoline vehicles. I need to identify three distinct and significant benefits. I will consider economic, environmental, and performance aspects.

1.  **Environmental Benefits**: EVs produce zero tailpipe emissions, which significantly reduces local air pollution (smog, particulate matter) compared to gasoline vehicles. This is a major advantage for public health and environmental quality, especially in urban areas.
2.  **Lower Running Costs**: EVs typically have lower 'fuel' costs (electricity vs. gasoline) per mile, especially when charging at home. Additionally, EVs have fewer moving parts than gasoline engines (no oil changes, spark plugs, complex transmissions, etc.), which generally leads to lower maintenance costs over the vehicle's lifespan.
3.  **Enhanced Driving Experience**: EVs offer instant torque from a standstill, resulting in quicker acceleration and a more responsive driving feel. They are also significantly quieter than gasoline vehicles due to the absence of an internal combustion engine, contributing to a smoother and more peaceful ride.

These three points cover environmental, economic, and performance advantages, which are key differentiating factors.

Answer: 1. Lower running costs (fuel and maintenance).
2. Environmental benefits (zero tailpipe emissions).
3. Enhanced driving experience (instant torque, quieter operation).
```


## **Few-Shot Implementation**

Now let's see the implementation of few-shot universal self consistency promtping technique using LangChain v1.0

```python
# ---------------------------------------------------------
# Few-Shot Universal Self-Consistency Prompting (USC)
# ---------------------------------------------------------

# pip install langchain langchain-google-genai pydantic

import os
import time
from pydantic import BaseModel, Field
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ---------------------------------------------------------
# 1. Set Gemini API key
# ---------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ---------------------------------------------------------
# 2. Define structured output schema
# ---------------------------------------------------------
class USCResponse(BaseModel):
    reasoning_chain: str = Field(..., description="Full chain-of-thought reasoning")
    answer: str = Field(..., description="Final concise answer")


parser = PydanticOutputParser(pydantic_object=USCResponse)


# ---------------------------------------------------------
# 3. Initialize Gemini model with sampling enabled
# ---------------------------------------------------------
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.8,
    top_k=40,
)


# ---------------------------------------------------------
# 4. Few-shot example (replaced with EV example)
# ---------------------------------------------------------
few_shot_example = """
Example Problem:
What are three advantages of electric vehicles over gasoline vehicles?

Example Chain-of-Thought:
Electric vehicles (EVs) offer several benefits compared to gasoline vehicles. 
First, EVs have lower operating costs because electricity is cheaper than gasoline. 
Second, they produce zero tailpipe emissions, which helps reduce air pollution. 
Third, EVs have fewer moving parts, which reduces maintenance requirements.

Example Final Answer:
Lower operating cost; zero tailpipe emissions; fewer moving parts.
"""


# ---------------------------------------------------------
# 5. Few-shot generation prompt
# ---------------------------------------------------------
generation_prompt_template = ChatPromptTemplate.from_template(
    """
You are a detailed step-by-step reasoning assistant.

Below is a worked example:
{few_shot_example}

Now use the same style of reasoning to answer the new question.

New Question:
{question}

Instructions:
- Provide a full chain-of-thought reasoning.
- Then give a concise final answer summarizing the key rules.
- Respond in this JSON format:
{format_instructions}
"""
)

generation_prompt = generation_prompt_template.partial(
    few_shot_example=few_shot_example,
    format_instructions=parser.get_format_instructions()
)

gen_chain = generation_prompt | model | parser


# ---------------------------------------------------------
# 6. Universal Self-Consistency Selection Prompt
# ---------------------------------------------------------
selection_prompt = ChatPromptTemplate.from_template(
    """
You are an evaluator assistant.

You are given multiple candidate responses to the same question.
Your task is to read ALL the responses and select the one that is
the most consistent, complete, and logically sound.

Question:
{question}

Candidate Responses:
{all_responses}

Instructions:
- Compare the reasoning across responses.
- Choose the single best response.
- Return ONLY the index number (1, 2, or 3).
- Do NOT explain your choice.

Return output in plain text containing ONLY the index number.
"""
)

selection_chain = selection_prompt | model


# ---------------------------------------------------------
# 7. Universal Self-Consistency function (n_samples=3)
# ---------------------------------------------------------
def universal_self_consistency(question: str, n_samples: int = 3):
    candidates = []

    # --- Stage 1: Generate candidate answers ---
    for _ in range(n_samples):
        out = gen_chain.invoke({"question": question})
        candidates.append(out)
        time.sleep(1)

    # Create formatted text block for evaluation
    formatted = ""
    for idx, c in enumerate(candidates, 1):
        formatted += f"\n[{idx}] Reasoning:\n{c.reasoning_chain}\nAnswer: {c.answer}\n"

    # --- Stage 2: LLM selects best answer ---
    chosen_idx = selection_chain.invoke(
        {"question": question, "all_responses": formatted}
    )
    chosen_idx = int(chosen_idx.content.strip())

    return candidates[chosen_idx - 1], candidates


# ---------------------------------------------------------
# 8. Run Few-shot Universal Self-Consistency
# ---------------------------------------------------------
question = "What are the most important rules for creating a strong password?"

best_output, all_outputs = universal_self_consistency(question, n_samples=3)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------
print("\n===== FINAL CHOSEN ANSWER =====")
print(best_output.answer)

print("\n===== ALL GENERATED CANDIDATES =====")
for i, out in enumerate(all_outputs, 1):
    print(f"\n--- Candidate {i} ---")
    print(out.reasoning_chain)
    print("Answer:", out.answer)

```

Here the output is
```

===== FINAL CHOSEN ANSWER =====
Length (12+ characters); Mix of character types (uppercase, lowercase, numbers, symbols); Avoid easily guessable information; Unique for each account.

===== ALL GENERATED CANDIDATES =====

--- Candidate 1 ---
Creating a strong password is crucial for online security. The most important rules revolve around making it difficult for others to guess or for automated tools to crack. 
First, the password should be sufficiently long, ideally 12 characters or more, as longer passwords significantly increase the number of possible combinations and thus the time it takes to crack them. 
Second, it must incorporate a variety of character types, including a mix of uppercase letters, lowercase letters, numbers, and special symbols (e.g., !, @, #, $). This diversity prevents simple dictionary attacks and brute-force attempts that target specific character sets. 
Third, it is vital to avoid using easily guessable information such as personal details (birthdays, names, pet names), sequential patterns (e.g., '123456', 'qwerty'), or common dictionary words, as these are frequently targeted. 
Fourth, each password should be unique for every account. Reusing passwords means that if one account is compromised, all other accounts using the same password become vulnerable. 
Finally, consider using a password manager to generate and store complex, unique passwords, as this helps enforce all these rules consistently.

Answer: Use a long password (12+ characters); include a mix of uppercase, lowercase, numbers, and symbols; avoid personal information and common words; use unique passwords for each account.

--- Candidate 2 ---
Creating a strong password is a fundamental aspect of digital security. First, the most crucial rule is to make the password long; experts generally recommend a minimum of 12-16 characters, as length significantly increases the computational effort required to crack it. Second, a strong password must incorporate a variety of character types, including uppercase letters, lowercase letters, numbers, and special symbols, which adds to its complexity and unpredictability. Third, it is vital to use a unique password for every different online account; reusing passwords means that if one service is breached, all other accounts using the same password become vulnerable. Fourth, users should avoid easily guessable information, such as personal details (like names, birthdays, or pet names), common words found in dictionaries, or simple sequential patterns (like '123456' or 'qwerty'). Finally, consider using a password manager to generate and store truly random and complex passwords, ensuring they are both strong and unique without needing to be memorized.

Answer: Use a long password (12+ characters); include a mix of uppercase, lowercase, numbers, and symbols; ensure it's unique for each account; and avoid personal information, common words, or simple patterns.

--- Candidate 3 ---
Creating a strong password is essential for digital security. First, the most crucial aspect is **length**. A strong password should be long, ideally 12 characters or more, as this significantly increases the number of possible combinations and makes it much harder for attackers to crack through brute-force methods. Second, it's vital to incorporate a **mix of character types**. This means using a combination of uppercase letters, lowercase letters, numbers, and special symbols (like !, @, #, $, %, etc.). This variety adds complexity and prevents simple dictionary or pattern-based attacks. Third, avoid using **easily guessable information**. This includes personal details (like names, birthdays, pet names), common words, sequential numbers (12345), or simple keyboard patterns (qwerty). Such passwords are often the first targets for attackers. Finally, ensure the password is **unique** for each account. Reusing passwords means that if one account is compromised, all other accounts using the same password become vulnerable.

Answer: Length (12+ characters); Mix of character types (uppercase, lowercase, numbers, symbols); Avoid easily guessable information; Unique for each account.
```
