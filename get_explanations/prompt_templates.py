import os
import json
from typing import List, Dict

BASE_DIR = os.path.dirname(__file__)


def create_explanation_generation_prompt(
        final_criteria_list: List[Dict[str, str]],
        language: str = "solidity"
) -> str:
    criteria_str = ""
    for i, criteria in enumerate(final_criteria_list):
        name = criteria.get('name', 'Unnamed Criteria')
        rationale = criteria.get('rationale', 'No rationale provided.')
        criteria_str += f"{i + 1}. **{name}**: {rationale}\n"

    json_keys = [f'    "{criteria.get("name")}": "Your detailed assessment for this criterion."'
                 for criteria in final_criteria_list]
    json_keys_str = ",\n".join(json_keys)

    prompt = f"""
You are a code analysis expert. Your task is to provide a detailed, multi-faceted analysis of a given piece of code based on a specific set of criteria.

**Please analyze the upcoming code snippet from the following {len(final_criteria_list)} perspectives:**

{criteria_str}
**Your Mission:**
For each criterion listed above, provide a concise and factual assessment based on the provided code.

**Output Format:**
Your entire output must be a single, well-formed JSON object. Do not add any text before or after the JSON. The JSON keys must exactly match the names of the criteria provided above.

The structure should be as follows:
```json
{{
{json_keys_str}
}}
```
"""
    return prompt.strip()


def format_perspectives_to_markdown_list(perspectives):
    markdown_lines = [
        f"- **{p['name']}**: {p['rationale']}"
        for p in perspectives
    ]

    return "\n".join(markdown_lines)


GET_PROMPT_SYSTEM = """You are an expert in program analysis. Your goal is to help me create a set of analysis perspectives for a new code classification task."""


def GET_PROMPT_USER(task_definition):
    return f"""**1. Definition of the New Code Classification Task:**
{task_definition}

**2. Existing General Perspectives:**
The system already uses these general perspectives for basic understanding. **Your new perspectives should not overlap with these**:

- **Basic Functionality Interpretation**: Explain the primary purpose and functionality of the code. What problem does it solve at a high level?
- **Logic and Flow Interpretation**: Describe the execution logic and control flow of the code. How does it achieve its purpose step-by-step, including key branches, loops, and decision points?
- **External Dependency Analysis**: Detail all interactions the code has with the outside world. What external functions, libraries, or APIs does it call? What external state (like databases, files, or contract storage) does it read from or write to?

**3. Your Mission:**
Generate a list of 10-15 **new, task-specific, and diagnostic perspectives** that are highly relevant for the defined task.

For each new perspective, provide:
- A short, unique `name`.
- A concise `rationale` explaining its importance for this specific task.

**4. Output Format:**
Your entire output must be a single JSON object enclosed in a Markdown code block. Do not add any text before or after the Markdown block. Follow this structure exactly:

```json
{{
  "specific_perspectives": [
    {{
      "name": "Example Perspective Name 1",
      "rationale": "Example rationale explaining why this perspective is crucial for the task."
    }},
    {{
      "name": "Example Perspective Name 2",
      "rationale": "Another example rationale."
    }}
  ]
}}
```
"""


FILTER_PROMPT1_SYSTEM = """You are a meticulous data curator specializing in semantic deduplication. Your task is to identify and filter out redundant perspectives from a list of candidates."""


def FILTER_PROMPT1_USER(general_criteria, specific_criteria):
    return f"""There are two sets of analysis perspectives:

**Set A: Existing General Perspectives (The reference set)**
{format_perspectives_to_markdown_list(general_criteria)}

**Set B: Candidate Specific Perspectives (The set to be filtered)**
{format_perspectives_to_markdown_list(specific_criteria)}

**Your Mission:**
Carefully compare each perspective in **Set B** against all perspectives in **Set A**. Identify any perspective from **Set B** that is semantically similar or largely overlaps in purpose with any perspective in **Set A**.

A perspective is considered "similar" if it addresses the same core question, analyzes the same aspect of the code, or produces a very similar type of explanation.

**Output Format:**
Your entire output must be a single JSON object. Do not add any text before or after the JSON.
- If you find one or more redundant perspectives in Set B, list their exact names in the `perspectives_to_remove` array.
- **If you find no redundancies and all candidate perspectives in Set B are unique, return an empty array `[]`.**

Follow this structure exactly:

```json
{{
  "perspectives_to_remove": [
    "Name of a redundant perspective from Set B",
    "Name of another redundant perspective from Set B"
  ]
}}
```

**Example of an empty output (if no redundancies are found):**

```json
{{
  "perspectives_to_remove": []
}}
```
"""


FILTER_PROMPT2_SYSTEM = """You are an expert conceptual synthesizer. Your task is to refine a list of analysis perspectives by merging those that are semantically similar or address the same underlying concept."""


def FILTER_PROMPT2_USER(filtered_criteria):
    return f"""Here is the list of perspectives to be refined:
---
**Perspectives to Refine:**
{format_perspectives_to_markdown_list(filtered_criteria)}
---

**Your Mission:**
1.  Carefully review the entire list.
2.  Identify groups of two or more perspectives that are closely related or overlap significantly in their goals.
3.  For each identified group, merge them into a **single, more comprehensive perspective**.
    -   Create a new, representative `name` for the merged perspective.
    -   Write a new `rationale` that synthesizes the core ideas of the original perspectives.
4.  Perspectives that are unique and distinct should be kept as they are.

**Output Format:**
Your entire output must be a single JSON object enclosed in a Markdown code block, containing the final, refined list of perspectives. The structure must be identical to the one specified below. Do not add any text before or after the Markdown block.

```json
{{
  "specific_perspectives": [
    {{
      "name": "Refined or Merged Perspective Name 1",
      "rationale": "The synthesized rationale for this perspective."
    }},
    {{
      "name": "A Unique Perspective That Was Kept",
      "rationale": "The original rationale for this unique perspective."
    }}
  ]
}}
```
"""


SCORING_PROMPT_SYSTEM = """You are a meticulous relevance evaluator. Your task is to assess how useful and relevant a specific analysis perspective is for classifying a given code snippet according to a defined task."""


def SCORING_PROMPT_USER(task_description, language, code, perspective_name, perspective_rationale):
    return f"""
**1. Task Definition:**
{task_description}

**2. Code Snippet to Analyze:**
```{language}
{code}
```

**3. Perspective to Evaluate:**
- **Name**: {perspective_name}
- **Rationale**: {perspective_rationale}

**4. Your Mission:**
Based on the **Task Definition**, evaluate how relevant and useful the **Perspective to Evaluate** is for analyzing the provided **Code Snippet**.

Please provide a numerical score from 1 to 5, where:
- **1**: Completely irrelevant. This perspective provides no useful information for this specific code.
- **2**: Mostly irrelevant. The connection is very weak or coincidental.
- **3**: Moderately relevant. It might provide some background context but is not a strong indicator.
- **4**: Highly relevant. This perspective directly addresses a key aspect of the code related to the task.
- **5**: Critically relevant. An analysis from this perspective is essential for making a correct classification for this code.

**5. Output Format:**
Your entire output must be a single JSON object containing only the relevance score. Do not add any text before or after the JSON.

```json
{{
  "relevance_score": <your score from 1 to 5>
}}
```
"""


TASK_DESC_PROMPT_SYSTEM = f"""You are an expert research assistant tasked with creating a formal and comprehensive definition for a new code classification task."""


def TASK_DESC_PROMPT_USER(task_name, samples_block):
    return f"""**1. Task Name:**
{task_name}

**2. Code Samples:**
{samples_block}

**3. Your Mission:**
Based on the provided task name and code samples, generate a formal task description by filling in the fields of the JSON object below.

**4. Output Format:**
Your entire output must be a single JSON object enclosed in a Markdown code block. Do not add any text before or after the Markdown block.

```json
{{
  "task_description": 
  {{
    "core_task": "Summarize the primary goal in one sentence. Mention all classification labels.",
    "context": "Infer the programming language, the granularity of analysis (e.g., function, class), and any specific frameworks.",
    "mechanism_and_impact": "This is the most important part. Describe what distinguishes the different labels. What are the common patterns or logic flaws you observed in the samples for each label? What are the potential consequences?",
    "non_goals": "Specify aspects that are likely irrelevant for this classification, such as code style or comments, unless they directly cause the core issue."
  }}
}}
```"""
