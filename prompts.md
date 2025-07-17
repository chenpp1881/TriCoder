# TriCoder-Prompts


# Complete Prompt Templates for TriCoder Implementation

This section presents all prompt templates used in implementing **TriCoder**, where content requiring substitution is indicated with square brackets `[]`. Note that all the prompts have been embedded in our reproducible package and can be directly invoked.


## Task-Specific Perspective Generation Prompt

The prompt template for generating task-specific perspectives ($\mathcal{P}_{\text{raw}}$) from classification task descriptions is as follows:

```
You are an expert in program analysis. Your goal is to help 
me create a set of analysis perspectives for a new code 
classification task.

**1. Definition of the New Code Classification Task:**
[Task Definition]

**2. Existing General Perspectives:**
The system already uses these general perspectives for 
basic understanding. **Your new perspectives should not 
overlap with these**:

- **Basic Functionality Interpretation**
- **Logic and Flow Interpretation**
- **External Dependency Analysis**

**3. Your Mission:**
Generate a list of 10-15 **new, task-specific, and 
diagnostic perspectives** that are highly relevant for the 
defined task.

For each new perspective, provide:
- A short, unique `name`.
- A concise `rationale` explaining its importance for this 
specific task.

**4. Output Format:**
Your entire output must be a single JSON object enclosed in 
a Markdown code block. Do not add any text before or after 
the Markdown block. Follow this structure exactly:

json
{
  "Task-specific Perspectives": [
    {
      "name": "Example Perspective Name 1",
      "rationale": "Example rationale explaining why this perspective is crucial for the task."
    },
    {
      "name": "Example Perspective Name 2",
      "rationale": "Another example rationale."
    }
  ]
}
```

## Perspective Deduplication Prompt

The prompt template for identifying and removing semantically redundant perspectives from the general and task-specific sets is as follows:

```
You are a meticulous data curator specializing in semantic 
deduplication. Your task is to identify and filter out 
redundant perspectives from a list of candidates.

There are two sets of analysis perspectives:

**Set A: Existing General Perspectives (Reference set)**
[Universal Perspectives]

**Set B: Candidate Specific Perspectives (The set to be filtered)**
[Task-specific Perspectives]

**Your Mission:**
Carefully compare each perspective in **Set B** against all 
perspectives in **Set A**. Identify any perspective from 
**Set B** that is semantically similar or largely overlaps 
in purpose with any perspective in **Set A**.

A perspective is considered "similar" if it addresses the
same core question, analyzes the same aspect of the code, or 
produces a very similar type of explanation.

**Output Format:**
Your entire output must be a single JSON object. Do not add 
any text before or after the JSON.

- If you find one or more redundant perspectives in Set B, 
  list their exact names in the `perspectives_to_remove` array.
- **If you find no redundancies, all candidate perspectives
  in Set B are unique, return an empty array `[]`.**

Follow this structure exactly:

json
{
  "perspectives_to_remove": [
    "Name of a redundant perspective from Set B",
    "Name of another redundant perspective from Set B"
  ]
}

**Example of an empty output (no redundancies found):**

json
{
  "perspectives_to_remove": []
}

```

## Perspective Consolidation Prompt

The prompt template for merging semantically similar perspectives into comprehensive, unified ones is as follows:

```
You are an expert conceptual synthesizer. Your task is 
to refine a list of analysis perspectives by merging 
those that are semantically similar or address the same 
underlying concept.

Here is the list of perspectives to be refined:
---
**Perspectives to Refine:**
[Task-specific Perspectives]
---

**Your Mission:**
1. Carefully review the entire list.
2. Identify groups of two or more perspectives that 
   are closely related or overlap significantly in their goals.
3. For each identified group, merge them into a 
   **single, more comprehensive perspective**:
   - Create a new, representative `name`.
   - Write a new `rationale` that synthesizes the core ideas.
4. Perspectives that are unique and distinct should be kept 
   as they are.

**Output Format:**
Your entire output must be a single JSON object enclosed in 
a Markdown code block. Do not add any text before or after 
the block. Use the following structure:

```json
{
  "Task-specific Perspectives": [
    {
      "name": "Refined or Merged Perspective Name 1",
      "rationale": "The synthesized rationale for this perspective."
    },
    {
      "name": "A Unique Perspective That Was Kept",
      "rationale": "The original rationale for this unique perspective."
    }
  ]
}
```

## Relevance Scoring Prompt

The prompt template for evaluating the relevance score of task-specific perspectives is as follows:

```
You are a meticulous relevance evaluator. Your task is to 
assess how useful and relevant a specific analysis 
perspective is for classifying a given code snippet 
according to a defined task.

**1. Task Definition:**
[Task Definition]

**2. Code Snippet to Analyze:**
[Language]
[Code]

**3. Perspective to Evaluate:**
- **Name**: [Perspective Name]
- **Rationale**: [Perspective Rationale]

**4. Your Mission:**
Based on the **Task Definition**, evaluate how relevant and
useful the **Perspective to Evaluate** is for analyzing the 
provided **Code Snippet**.

Please provide a numerical score from 1 to 5:

- **1**: Completely irrelevant
- **2**: Mostly irrelevant
- **3**: Moderately relevant
- **4**: Highly relevant
- **5**: Critically relevant

**5. Output Format:**
Return a single JSON object containing only the relevance score:

```json
{
  "relevance_score": <your score from 1 to 5>
}
```

## Multi-Perspective Code Analysis Prompt

The prompt template for generating multi-perspective explanations of code snippets is as follows:

```
You are a code analysis expert. Your task is to provide a 
detailed, multi-faceted analysis of a given piece of code
based on a specific set of criteria.

**Please analyze the upcoming code snippet from the 
following [N] perspectives:**

[Universal Perspectives]  
[Task-specific Perspectives]

**Your Mission:**
For each criterion listed above, provide a concise and 
factual assessment based on the provided code.

**Output Format:**
Return a single, well-formed JSON object. Do not add any 
text before or after the JSON. The keys must match the 
perspective names exactly:

```json
{
  "[Perspective Name 1]": "Your detailed assessment for this criterion.",
  "[Perspective Name 2]": "Your detailed assessment for this criterion.",
  ...
}
```
