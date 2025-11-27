# **Self Refine Prompting**

Authored by Kalyan KS. You can follow me on [Twitter](https://x.com/kalyan_kpl) and [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/) for latest LLM, RAG and Agent updates.

## **Overview**

Self-Refine Prompting is an iterative reasoning technique in which a model improves its own output through a repeated cycle of generation → feedback → refinement.

Instead of producing a single answer in one attempt, the model first drafts an initial solution, then evaluates it, identifies flaws or opportunities for improvement, and finally produces a refined version. This loop can repeat several times until a high-quality final output is reached.

Just as a human writer drafts a paragraph, rereads it, notices issues, and revises it, the model engages in self-reflection to improve accuracy, clarity, and quality.


![Self Refine Prompting](images/1-self-refine-prompt.jpg)

Figure from [Self Refine Prompting](https://arxiv.org/abs/2303.17651) paper. 


## **Prompt Template**

Here is the initial draft prompt template for self refine prompting.

```
You are an expert Python developer.

Write the first draft solution to the task below, without feedback or refinement.
Focus only on producing an initial attempt.

Task:
{task}
```
Here is the feedback prompt template for self refine prompting.

```
You are an expert code reviewer.

Given the initial draft below, generate **specific and actionable feedback**.
Your feedback MUST identify:
- What is missing
- What is incorrect
- What can be improved
- Why the improvement is important

Do NOT rewrite the answer. Only critique it.

Task:
{task}

Initial Draft:
{draft}
```

Here is the refinement prompt template for self refine prompting.

```
You are an expert Python developer.

Refine the initial draft by applying the feedback.
Your refined version MUST:
- Correct errors
- Address missing logic
- Improve quality, clarity, and reliability
- Follow best Python practices

Task:
{task}

Initial Draft:
{draft}

Feedback:
{feedback}

Now produce the improved answer.
```

## **Stay Updated with Generative AI, LLMs, Agents and RAG**

Join 🚀 [**AIxFunda** free newsletter](https://aixfunda.substack.com/) to get *latest updates* and *interesting tutorials* related to Generative AI, LLMs, Agents and RAG. 
- ✨ Weekly GenAI updates
- 📄 Weekly LLM, Agents and RAG research paper updates
- 📝 1 fresh blog post on an interesting topic every week

## **Implementation**

Now let's see the implementation of self refine promtping technique (without multi-loop iteration) using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# 1. Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ----------------------------------------------------------
# 2. Define Structured Output Models for Self-Refine
# ----------------------------------------------------------

class InitialDraft(BaseModel):
    draft: str = Field(..., description="The model's initial attempt at the solution")


class Feedback(BaseModel):
    feedback: str = Field(..., description="Actionable and specific feedback describing issues and improvements")


class RefinedOutput(BaseModel):
    refined_answer: str = Field(..., description="Improved solution incorporating the feedback")


initial_parser = PydanticOutputParser(pydantic_object=InitialDraft)
feedback_parser = PydanticOutputParser(pydantic_object=Feedback)
refine_parser = PydanticOutputParser(pydantic_object=RefinedOutput)


# ----------------------------------------------------------
# 3. Initialize Gemini model (gemini-2.5-flash)
# ----------------------------------------------------------

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)


# ----------------------------------------------------------
# 4. Prompt Templates for INITIAL → FEEDBACK → REFINE
# ----------------------------------------------------------

# 4.1 Initial Draft Prompt
initial_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert Python developer.

Write the first draft solution to the task below, without feedback or refinement.
Focus only on producing an initial attempt.

Task:
{task}

Provide the output in this JSON format:
{format_instructions}
"""
)

initial_prompt = initial_prompt_template.partial(
    format_instructions=initial_parser.get_format_instructions()
)


# 4.2 Feedback Prompt
feedback_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert code reviewer.

Given the initial draft below, generate **specific and actionable feedback**.
Your feedback MUST identify:
- What is missing
- What is incorrect
- What can be improved
- Why the improvement is important

Do NOT rewrite the answer. Only critique it.

Task:
{task}

Initial Draft:
{draft}

Provide your feedback in this JSON format:
{format_instructions}
"""
)

feedback_prompt = feedback_prompt_template.partial(
    format_instructions=feedback_parser.get_format_instructions()
)


# 4.3 Refinement Prompt
refine_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert Python developer.

Refine the initial draft by applying the feedback.
Your refined version MUST:
- Correct errors
- Address missing logic
- Improve quality, clarity, and reliability
- Follow best Python practices

Task:
{task}

Initial Draft:
{draft}

Feedback:
{feedback}

Now produce the improved answer.

Provide the refined output in this JSON format:
{format_instructions}
"""
)

refine_prompt = refine_prompt_template.partial(
    format_instructions=refine_parser.get_format_instructions()
)


# ----------------------------------------------------------
# 5. Build LCEL Chains
# ----------------------------------------------------------

initial_chain = initial_prompt | model | initial_parser
feedback_chain = feedback_prompt | model | feedback_parser
refine_chain = refine_prompt | model | refine_parser


# ----------------------------------------------------------
# 6. Run Self-Refine on the User's Example Task
# ----------------------------------------------------------

task = "Write a Python function calculate_average that takes a list of numbers and returns the average."

# Phase 1 — Initial Draft
initial_result = initial_chain.invoke({"task": task})
print("\n--- INITIAL DRAFT ---\n")
print(initial_result.draft)

# Phase 2 — Feedback
feedback_result = feedback_chain.invoke({
    "task": task,
    "draft": initial_result.draft
})
print("\n--- FEEDBACK ---\n")
print(feedback_result.feedback)

# Phase 3 — Refinement
refined_result = refine_chain.invoke({
    "task": task,
    "draft": initial_result.draft,
    "feedback": feedback_result.feedback
})
print("\n--- REFINED SOLUTION ---\n")
print(refined_result.refined_answer)
```

Here the output is

```

--- INITIAL DRAFT ---

def calculate_average(numbers):
    """
    Calculates the average of a list of numbers.

    Args:
        numbers (list): A list of numbers (integers or floats).

    Returns:
        float: The average of the numbers in the list.

    Raises:
        ValueError: If the input list is empty.
    """
    if not numbers:
        raise ValueError("Input list cannot be empty.")
    
    total_sum = sum(numbers)
    count = len(numbers)
    return total_sum / count

--- FEEDBACK ---

### What is missing:
1.  **Return Type Hint:** The function signature is missing a return type hint (`-> float`), which is specified in the docstring.

### What is incorrect:
1.  The core logic for calculating the average and handling an empty list is correct.

### What can be improved:
1.  **Add Return Type Hint:** Explicitly add `-> float` to the function signature.
2.  **Input Validation for Element Types:** While the docstring specifies 'A list of numbers (integers or floats)', the current implementation relies on `sum()` to raise a `TypeError` if non-numeric elements are present. This could be improved by adding explicit validation for the types of elements within the `numbers` list.
3.  **More Specific Error Handling for Non-Numeric Types:** Instead of letting `sum()` raise a generic `TypeError`, the function could catch this or perform checks to raise a more specific `TypeError` or `ValueError` if the list contains non-numeric items.

### Why the improvement is important:
1.  **Return Type Hint:** Adding `-> float` to the signature improves code readability, maintainability, and enables static analysis tools (like MyPy) to catch potential type-related bugs at development time, making the function's contract clearer and more robust.
2.  **Input Validation for Element Types:** Explicitly validating that all elements in the `numbers` list are indeed numbers (integers or floats) ensures the function adheres strictly to its documented input contract. This prevents unexpected runtime errors from internal Python functions (`sum()` in this case) and allows the function to provide more controlled and user-friendly error messages.
3.  **More Specific Error Handling:** Providing a custom error message when non-numeric types are encountered makes debugging easier for the caller. Instead of a generic `TypeError` from `sum()`, a message like "Input list must contain only numbers" would clearly indicate the problem, improving the user experience and the robustness of the function.

--- REFINED SOLUTION ---

def calculate_average(numbers: list) -> float:
    """
    Calculates the average of a list of numbers.

    Args:
        numbers (list): A list of numbers (integers or floats).

    Returns:
        float: The average of the numbers in the list.

    Raises:
        ValueError: If the input list is empty.
        TypeError: If the input list contains non-numeric elements.
    """
    if not numbers:
        raise ValueError("Input list cannot be empty.")

    # Validate that all elements in the list are numbers (int or float)
    for num in numbers:
        if not isinstance(num, (int, float)):
            raise TypeError("All elements in the input list must be numbers (int or float).")
    
    total_sum = sum(numbers)
    count = len(numbers)
    return total_sum / count
```


## **Implemntation (Multi-loop)**

Now let's see the implementation of self refine promtping technique with multi-loop iteration using LangChain v1.0

```python
# pip install langchain langchain-google-genai pydantic

import os
import time
from google.colab import userdata
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# ----------------------------------------------------------
# 2. Define Structured Output Models
# ----------------------------------------------------------

class InitialDraft(BaseModel):
    draft: str = Field(..., description="The model's initial attempt at the solution")

class Feedback(BaseModel):
    feedback: str = Field(..., description="Specific, actionable feedback. If no issues, must include the phrase 'no issues'.")

class RefinedOutput(BaseModel):
    refined_answer: str = Field(..., description="Improved answer incorporating the feedback")


initial_parser = PydanticOutputParser(pydantic_object=InitialDraft)
feedback_parser = PydanticOutputParser(pydantic_object=Feedback)
refine_parser = PydanticOutputParser(pydantic_object=RefinedOutput)


# ----------------------------------------------------------
# 3. Initialize Gemini model
# ----------------------------------------------------------

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0
)


# ----------------------------------------------------------
# 4. Prompt Templates
# ----------------------------------------------------------

# 4.1 Initial Draft Prompt
initial_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert Python developer.

Write the FIRST DRAFT solution to the task below.
Do NOT critique or refine it yet.

Task:
{task}

Output format:
{format_instructions}
"""
)

initial_prompt = initial_prompt_template.partial(
    format_instructions=initial_parser.get_format_instructions()
)


# 4.2 Feedback Prompt
feedback_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert code reviewer.

Carefully analyze the initial or refined draft.
Provide feedback that is:

- Specific
- Actionable
- Mentioning what to fix and why

If the answer is already correct, complete, and high-quality,
write feedback that **explicitly contains the phrase "no issues"**.

Task:
{task}

Draft Under Review:
{draft}

Output format:
{format_instructions}
"""
)

feedback_prompt = feedback_prompt_template.partial(
    format_instructions=feedback_parser.get_format_instructions()
)


# 4.3 Refinement Prompt
refine_prompt_template = ChatPromptTemplate.from_template(
    """
You are an expert Python developer.

Refine the draft by applying the feedback.
Improve correctness, clarity, robustness, and Python best practices.

Task:
{task}

Draft:
{draft}

Feedback:
{feedback}

Output format:
{format_instructions}
"""
)

refine_prompt = refine_prompt_template.partial(
    format_instructions=refine_parser.get_format_instructions()
)


# ----------------------------------------------------------
# 5. Build LCEL Chains
# ----------------------------------------------------------

initial_chain = initial_prompt | model | initial_parser
feedback_chain = feedback_prompt | model | feedback_parser
refine_chain = refine_prompt | model | refine_parser


# ----------------------------------------------------------
# 6. Multi-Iteration Self-Refine Loop (Stop When “no issues”)
# ----------------------------------------------------------

task = "Write a Python function calculate_average that takes a list of numbers and returns the average."

MAX_ITER = 3    # upper limit for safety

# Phase 1 — Generate initial draft
draft_result = initial_chain.invoke({"task": task})
current_draft = draft_result.draft

print("\n=== INITIAL DRAFT ===\n")
print(current_draft)

# Phase 2 — Iterative refine loop
for iteration in range(MAX_ITER):
    print(f"\n=== FEEDBACK ROUND {iteration} ===\n")

    # Generate feedback
    fb_result = feedback_chain.invoke({
        "task": task,
        "draft": current_draft
    })

    feedback = fb_result.feedback
    print(feedback)

    # Stop condition: feedback contains "no issues"
    if "no issues" in feedback.lower():
        print("\nStopping refinement: feedback reports 'no issues'.")
        break

		time.sleep(1)
    # Apply refinement
    refine_result = refine_chain.invoke({
        "task": task,
        "draft": current_draft,
        "feedback": feedback
    })

    current_draft = refine_result.refined_answer

    print(f"\n=== REFINED DRAFT {iteration} ===\n")
    print(current_draft)


print("\n\n=== FINAL OUTPUT AFTER SELF-REFINE ===\n")
print(current_draft)
```

Here the output is
```
=== INITIAL DRAFT ===

def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    count = len(numbers)
    return total / count

=== FEEDBACK ROUND 0 ===

The provided `calculate_average` function is well-written, efficient, and correctly implements the task. It handles the edge case of an empty list gracefully by returning 0, which is a reasonable convention for an average of an empty set. The use of built-in `sum()` and `len()` functions is Pythonic and efficient. There are no issues.

Stopping refinement: feedback reports 'no issues'.


=== FINAL OUTPUT AFTER SELF-REFINE ===

def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    count = len(numbers)
    return total / count
```