from typing import Any, Dict, List, Optional, Tuple
from utils.utils import sample_examples


skill_attribute_prompt = """
You are an expert Computer Science professor and a seasoned competitive programming coach. 
Your task is to analyze the provided programming problem and its reference solutions to identify:
1. The **ATOMIC** algorithmic skills needed to solve the problem
2. The **complexity attributes** - all characteristics that affect the problem's complexity/difficulty or could be varied to create mutations

Your analysis must be precise, accurate, and adhere to the standardized terminology of the field.

## Part 1: Skill Identification
1. Holistic Analysis: You must consider both the problem statement and the provided solutions.
    - The problem statement provides context and hints at the required complexity.
    - The solution code provides the definitive implementation, revealing the exact algorithms skills used.
    - If multiple solutions exist, include the union of distinct skills they use. Do not infer skills not evidenced by the code.
2. Skills
    - Identify the single most important and necessary concept that the problem is testing. Then, break down the concept into secondary or sub-level techniques/skills of the concept required to solve the problem.
    - Report only algorithmic programming-level techniques/data-structure operations (e.g., binary_search, two_pointers, monotonic_stack, dijkstra, fenwick_tree, prefix_sum, etc.).
    - Avoid overly generic terms (e.g., "logic," "math") or trivial implementation details (e.g., "variables," "loops").
    - Skills must be ATOMIC: a single, independent technique—not a composite or pipeline.
        - Split composites into primitives (e.g., segment_tree_with_lazy_propagation → segment_tree, lazy_propagation; binary_search_on_answer → parametric_search; etc.).
    - Names must be **lower_snake_case**, descriptive, and canonical (no synonyms, no duplicates).
    - Provide a brief, precise description of the skill.

## Part 2: Complexity Attributes

Analyze the solution and identify **every attribute that contributes to the problem's complexity or could be modified to create variations**. Think like a problem setter who needs to create variants of the problem with varying complexities.

### Discovery Process:
For each major component of the solution (input/output, loops, data structures, operations, conditions, etc.), ask:
- What choices were made here?
- What constraints or bounds exist?
- What could be different while maintaining the core algorithm (skill)?

### Categories to Explore (non-exhaustive):

**Quantitative Attributes** - Numeric values that affect complexity:
- Input size constraints (array length, string length, matrix dimensions)
- Value ranges (minimum/maximum values for integers, characters used)
- Iteration bounds (loop limits, recursion depth)
- Precision requirements (decimal places, modulo values)
- Count limits (number of operations allowed, query limits)

**Structural Attributes** - How the solution is organized:
- Data structure choices (array vs linked list, set vs map, etc.)
- Traversal patterns (left-to-right, inside-out, breadth-first vs depth-first, etc.)
- Processing order (sorted vs unsorted, online vs offline, etc.)
- Storage strategy (in-place vs auxiliary space, etc.)

**Operational Attributes** - What operations are performed:
- Core operations (addition vs multiplication, min vs max, etc.)
- Comparison types (strict vs non-strict inequality, etc.)
- Combination methods (sum, product, XOR, concatenation)
- Update strategies (increment, replace, accumulate, etc.)

**Logical Attributes** - Decision-making patterns:
- Branching conditions (what triggers different paths, etc.)
- Early termination conditions (when to stop)
- Selection criteria (how elements are chosen, etc.)
- Validation rules (what makes a solution valid)

**Complexity Factors** - What makes this instance hard:
- Number of nested structures
- Dependencies between computations
- State tracking requirements
- Edge case handling needs

### Extraction Guidelines:
For each complexity attribute you identify, provide:
1. **Attribute name**: A descriptive **lower_snake_case** identifier
2. **Description**: A clear definition of what this attribute represents. How varying this attribute affects problem difficulty.

## Part 3:Output Format
Your final response **must be valid JSON** exactly matching the schema below. Please output **ONLY** the JSON object. Do not include any extra things.
JSON SCHEMA TO FOLLOW:
{{
    "skills": {{
        <name of the skill>: <description of the skill>,
        ...
    }},
    "attributes": {{
        <name of the attribute>: <description of the attribute>,
        ...
    }}
}}

## Input
Problem:
{problem}

Reference Code Solution:
{code_solution}
""".strip()


cluster_skill_prompt = """
You are an expert computer science curriculum designer with extensive experience in creating structured learning paths for algorithmic problem-solving.

Your goal is to take a large, potentially redundant list of skills and perform a two-step refinement process:
1.  **Deduplicate**: Merge overlapping or similar skills into a set of canonical, non-overlapping skills with clear, consolidated descriptions.
2.  **Categorize**: Group the resulting canonical skills into broader categories based on primary algorithmic concepts or data-structure families.

## Input
You will be given a list of skills, where each skill has a `skill` name and a `description`.
{skill_list}

## Your Task
### Step 1: Deduplicate into Canonical Skills (Mental Step)
- First, mentally merge all synonymous, overlapping, or related skills from the input list into a single, canonical skill.
- If a skill from the input list is already distinct and does not overlap with any others, treat it as a canonical skill on its own; no deduplication is needed for it.
- For each canonical skill, define a standard `skill` name (in `lower_snake_case`) and write a new, concise `description` that synthesizes the core concept. For example, `array_sorting` and `custom_sorting_logic` should be merged into a single canonical skill called `sorting`.

### Step 2: Group Canonical Skills into Categories
- Now, take the set of canonical skills you just defined.
- Group these skills into high-level categories (e.g., "graph_algorithms", "dynamic_programming", "greedy_algorithms").
- The final output should be a list of these categories, each containing the relevant canonical skills you created.

## Output
Your final response **must be a list of valid JSON objects** exactly matching the schema below.
Please output **ONLY** the list of JSON objects. Do not include any extra things.

List of JSON SCHEMA TO FOLLOW:
{{
    "category": <name of the category>,
    "members": [
        {{
            "skill": <name of the skill>,
            "description": <description of the skill>,
        }},
        ...
    ]
}},
...
""".strip()

cluster_attribute_prompt = """
You are an expert computer science curriculum designer with extensive experience in creating structured learning paths for algorithmic problem-solving.

Your goal is to take a large, potentially redundant list of attributes that affect the complexity of a coding problem and group overlapping or similar attributes ubti a set of canonical, non-overlapping attributes with clear, consolidated descriptions.

## Input
You will be given a list of complexity attributes, where each attribute has a attribute name and a description of its definition and impact on problem complexity.
{attribute_list}

## Clustering Requirements

### Step 1: Group Similar Concepts
- Identify attributes that represent the same underlying complexity dimension
- Look for semantic equivalence, not just naming similarities
- Consider whether attributes would be varied together when creating problem mutations

### Step 2: Resolve Overlaps
- When attributes partially overlap, determine if they should merge or remain distinct
- Preserve distinctions only if they represent independently variable complexity dimensions
- Eliminate redundancy while maintaining coverage

### Step 3: Create Canonical Attributes
For each unified concept:
- Choose the most general and widely-applicable name (in `lower_snake_case`)
- Write a description that:
  * Defines what the attribute controls
  * Explains its impact on complexity when varied

### Quality Criteria
- Each canonical attribute should be independently mutable
- No two attributes should be redundant or co-dependent
- Names should be intuitive and follow standard CS terminology

## Output Format
Your final response must be a list of valid JSON objects exactly matching the schema below.
Please output ONLY the list of JSON objects. Do not include any extra things.

List of JSON SCHEMA TO FOLLOW:
{{
    <name of the attribute>: <description of the attribute>,
}},
...
""".strip()


skill_reflection_prompt = """
You are an expert Computer Science professor and a seasoned competitive programming coach. 
## Task: Identify the necessary algorithmic skills used in the provided Python code snippet.

You are given a list of algorithmic skills and a Python code snippet. You need to identify all the algorithmic skills used in the code snippet.

## Requirements
- Review the provided skill list and the code snippet carefully and thoroughly.
- Use only the skills listed in the provided skill set. Avoid creating new skills or altering the existing skill names.

Each code snippet may have multiple skills used in it but there is only one core testing skill that is the most important and necessary. 
You need to first detect the core testing skill, then continue to identify the other necessary skills used in the code snippet.
If there is no other skills except the core testing skill, then the "other_skills" should be an empty list.

## Output Format
Your final response **must be valid JSON** exactly matching the schema below. Please output **ONLY** the JSON object. Do not include any extra things.
JSON SCHEMA TO FOLLOW:
{{
    "main_skill": <name of the skill>,
    "other_skills": [
        <name of the skill>,
        ...
    ]
}}

## Input
Skill List:
{skill_list}

Code Snippet:
{code_snippet}
""".strip()


code_input_prompt = """
## Task: Create a New Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input

{failure_info}

Skills:
{skill_str}

### Code Requirements:
- The provided skill(s) should be the main testing concept of the code snippet.
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi-step reasoning.
- Avoid unnecessary function nesting; use simple, readable code structure when possible. Only use nested functions if it is natural for the algorithm
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{'age': 20, 'city': 'New York'}}
```

### Evaluation Criteria:
- Relevance (High Priority), the task should mainly focus on examining the ability of the test subject to understand / reason / apply the skill(s).
- Executability, your code should be executable given your input
- Difficulty in reverse-engineering your ```input``` from 1) your ```python``` code and 2) the deterministic ```output``` that will be obtained from your ```input```. Focus on either algorithmic reasoning or logic complexity. For example, you can define complex data structure classes and operate on them like trees, heaps, stacks, queues, graphs, etc, or use complex control flow, dynamic programming, recursions, divide and conquer, greedy, backtracking, etc
- Creativity, the code needs to be sufficiently different from the provided reference snippets
- Restricted usage of certain keywords and packages, you are not allowed to use the following words in any form, even in comments: <|BANNED_KEYWORDS|>

First, carefully devise a clear plan: e.g. how to make the skill(s) the main testing concept of the task, identify how your snippet will be challenging and creative. If there is previous failure information, think about how to resolve the failure and create an acceptable code snippet. Then, write the final code snippet and its input.
""".strip()

code_input_iterative_prompt = """
## Task: Create a New Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input

{failure_info}

Skills:
{skill_str}

### Previous Synthesis Context:
The code snippet provided below is created one iteration earlier for the skill above, plus current model's performance on this snippet. The performance is defined as the fraction of correct solutions over 10 independent solving attempts on the synthesized task (e.g., 3 correct solutions out of 10 attempts results in 0.3).
Code Snippet:
{prev_code}
Input:
{prev_input}
Performance:
{prev_performance}

You need to take this context into account while creating the new code snippet:
- Adjust difficulty based on previous performance: if high (>0.8), increase complexity and reasoning requirements.
- Ensure the new snippet is distinct from the previous one to maximize diversity.

### Code Requirements:
- The provided skill(s) should be the main testing concept of the code snippet.
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi-step reasoning.
- Avoid unnecessary function nesting; use simple, readable code structure when possible. Only use nested functions if it is natural for the algorithm
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{'age': 20, 'city': 'New York'}}
```

### Evaluation Criteria:
- Relevance (High Priority), the task should mainly focus on examining the ability of the test subject to understand / reason / apply the skill(s).
- Executability, your code should be executable given your input
- Difficulty in reverse-engineering your ```input``` from 1) your ```python``` code and 2) the deterministic ```output``` that will be obtained from your ```input```. Focus on either algorithmic reasoning or logic complexity. For example, you can define complex data structure classes and operate on them like trees, heaps, stacks, queues, graphs, etc, or use complex control flow, dynamic programming, recursions, divide and conquer, greedy, backtracking, etc
- Creativity, the code needs to be sufficiently different from the provided previous synthesized snippets
- Restricted usage of certain keywords and packages, you are not allowed to use the following words in any form, even in comments: <|BANNED_KEYWORDS|>

First, carefully devise a clear plan: e.g. how to make the skill(s) the main testing concept of the task, identify how your snippet will be challenging and creative. If there is previous failure information, think about how to resolve the failure and create an acceptable code snippet. Then, write the final code snippet and its input.
""".strip()

code_input_mutation_prompt = """
## Task: Create Multiple Variants of an Original Task by Systematically Applying Complexity Attributes to Increase Complexity and Reasoning Requirements.

Original task is a code reasoning deduction task that demands deep algorithmic reasoning to recover the hidden input from the provided output and Python code snippet.
You will be provided with:
- Original task: one Python code snippet and one output.
- Skill: the algorithmic skill(s) that the original task mainly tests.
- Complexity attributes: a list of complexity attributes that you can consider to increase the complexity and reasoning requirements of the original task. Be aware that not all attributes are applicable to the original task. You need to first determine which attributes could be used.

## Your Mission:

### Step 1: Analyze the Original Task
Examine the provided task and identify:
- Current complexity level for each **applicable** attribute
- Potential for complexity enhancement in each dimension

### Step 2: Generate Diverse Variants
Create **at least 10 different variants** via the following approaches:
- Single-attribute mutations: Enhance only one complexity dimension
- Multi-attribute mutations: Modify multiple but not all attributes at the same time
- Comprehensive mutations: Adjust all applicable attributes together

### Requirements for Each Variant:

#### Code Requirements:
- The provided skill(s) should be the main testing concept of the code snippet.
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi-step reasoning.
- Avoid unnecessary function nesting; use simple, readable code structure when possible. Only use nested functions if it is natural for the algorithm
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
#### Input Requirements:
- Provide exactly one hidden test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

#### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

#### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{{{'age': 20, 'city': 'New York'}}}}
```

#### Complexity Enhancements:
- Preserve the core algorithmic skill being tested
- Consider the original task's difficulty (pass rate) when increasing complexity. The variants should not be too easy (pass rate = 100%) or too hard (pass rate = 0%).
- Each variant should be meaningfully different from others
- Maintain solvability while increasing complexity
- Output the complexity attributes that you considered for each variant and one concise description of how you increase the complexity by each attribute and why it is more complex than the original task.

### Output Format:
Your final response **must be valid JSON** exactly matching the schema below. Please output **ONLY** the JSON object. Do not include any extra things.
JSON SCHEMA TO FOLLOW:
{{
    "variant_1": {{
        "complexity_attributes": [
            <name of the complexity attribute being considered>,
            ...
        ],
        "description": <description of how variant_1 is more complex than the original task by the complexity attributes you considered>,
        "code": "```python\n<code snippet of variant_1>\n```",
        "input": "```input\n<input of variant_1>\n```",
    }},
    "variant_2": ...,
    ...
}}

### Generation Strategy:

1. **Breadth First**: Generate variants that explore different complexity attributes before combining them
2. **Difficulty Progression**: Include variants ranging from moderate increases to significant complexity jumps
3. **Reasoning Diversity**: Create variants that fail for different reasons (timeout, overflow, logic errors, edge cases)

### Evaluation Criteria:
- Relevance to core algorithmic skills. Each variant should test the same core skill.
- Complexity increases should be meaningful, not artificial.
- Variants should explore different failure modes and challenge different aspects of problem-solving.
- Executability, your code should be executable given your input.

Remember: The goal is to create a rich set of practice problems that gradually build expertise by challenging different aspects of the skill through varied complexity enhancements.

## Original Task:
Code:
{code}

Input:
{input}

## Skill:
{skill}

## Complexity Attributes:
{complexity_attributes}
""".strip()

code_input_crossover_prompt = """
## TASK: Create a New Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input and with Novel Skill Combination

### Available Information:

**Skill Pool:**
{skill_pool}
- A comprehensive list of algorithmic coding skills along with their descriptions that you can choose from to create new combinations

**Target Core Skill:**
{target_skill}
- The primary skill that must be the central focus of your code snippet along with its description

**Existing Skill Combinations:**
{existing_combinations}
- Previously created combinations of the target skill with other skills

### Your Task:
You need to create a novel crossover by combining the target skill with other compatible skills from the skill pool. This requires:
1. Examining existing combinations to understand what has already been done
2. Identifying skills from the pool that would naturally work with the target skill but haven't been combined yet
3. Creating a code snippet where all chosen skills are essential and interconnected to hide the true input yet still produce the provided output

The key challenge is to ensure your skill combination is both NEW (not in existing combinations) and MEANINGFUL (skills genuinely complement each other rather than being artificially forced together).

### Code Requirements:
- The target skill MUST be the central concept, with other skills naturally supporting or enhancing it
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- The snippet should demonstrate clear interdependence between all skills in the combination (not just using skills in isolation)
- Require state tracking across multiple data transformations, ensuring multi-step reasoning
- Avoid unnecessary function nesting; use simple, readable code structure when possible
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`
{remove_input_from_snippet_prompt}{remove_after_return_prompt}

### Skill Combination Requirements:
- Your new combination must include the target skill plus at least one other skill from the skill pool
- The combination should be meaningfully different from all existing combinations
- Skills should work together synergistically, not just appear sequentially
- Justify why your chosen skills complement each other naturally and how they work together in your code snippet to obfuscate the hidden input

### Input Requirements:
- Provide exactly one hidden test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

### Output Format:
Your final response **must be valid JSON** exactly matching the schema below. Please output **ONLY** the JSON object. Do not include any extra things.
JSON SCHEMA TO FOLLOW:
{{
    "skill_combination": [<target_skill>, <new_combined_skill_1>, <new_combined_skill_2>, ...],
    "crossover_justification": <brief explanation of why and how these skills work well together>,
    "code": "```python\n<code snippet of the new skill combination>\n```",
    "input": "```input\n<input of the new skill combination>\n```",
}}

### Evaluation Criteria:
- Novelty (Critical): The skill combination must be different from all existing combinations
- Integration (High Priority): All skills must work together meaningfully, not just appear independently
- Target Skill Focus: The target skill should be the dominant concept being tested
- Executability: Your code should run successfully with the provided input
- Difficulty: The task should require understanding the interaction between skills, not just individual skill mastery, to reverse-engineer the hidden input
- Creativity: The scenario should be distinct from existing code snippets
- Complexity: Focus on algorithmic reasoning or logic complexity (e.g., complex data structures, control flow, dynamic programming, recursion)
- Restricted Keywords: You cannot use the following words in any form: <|BANNED_KEYWORDS|>

### Process:
- First, analyze the existing combinations to understand patterns and identify gaps
- Second, propose your new skill combination with clear justification
- Third, devise a plan for how these skills will interact in your code to conceal the input
- Fourth, implement the code snippet ensuring all skills are essential to the solution and following the code requirements
- Finally, provide the hidden input that exercises the full combination

If there is previous failure information, address those issues explicitly in your new attempt.
"""


code_input_curriculum_prompt = """
You are a senior mentor specializing in algorithmic programming education.

## Task
Given a student's performance profile on code abduction tasks, create **five** new Python code snippets, each with a matching test input, that target the student's identified weaknesses. The three tasks should address different weaknesses or explore the same weakness from different angles.

**Code Abduction Task Definition**: The student is given a code snippet and a output, then must deduce the correct input through reasoning (without executing the code).

## Provided Information
1. Student Performance Profile
- Contains tasks across three difficulty levels: **easy**, **medium**, and **hard**.
- Each level includes 10 example tasks, each annotated with:
  - **Student score**: Average accuracy across 10 attempts (range: 0.0 to 1.0).
  - **Tested skills**: The algorithmic concepts required to solve the task.
  - **Category**: The broader topic area of the task.
- Tasks are randomly sampled from the student's full performance history.
2. Categorized Algorithmic Skills
- A taxonomy of algorithmic skills, each with a description.
- The student should progressively master these skills through targeted practice.

## Your Objectives
1. **Analyze** the performance profile to identify the student's weaknesses.
2. **Create five** new code snippets, each with one matching test input, that address the identified weaknesses.
3. **Specify** the primary skills tested by each new task (must be drawn from the provided skill taxonomy).

## Analysis Guidelines
- Examine each example task carefully to identify patterns in the student's reasoning deficiencies.
- Prioritize skills where the student shows **consistent difficulty**, especially across "medium" and "hard" tasks.
- Favor weaknesses that appear across **multiple tasks or difficulty levels** over isolated failures.
- When uncertain, prioritize **algorithmic reasoning and complexity gaps** over minor accuracy issues.
- Ensure the new task's difficulty is appropriate: challenging enough to address weaknesses, but achievable with effort.
- Select **atomic skills** (i.e., specific, indivisible concepts) from the skill taxonomy.
- Ensure **novelty**: the new task should differ meaningfully from the sampled examples.
- Ensure **diversity**: the five tasks should vary in structure, skills tested, or approach.

## Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi-step reasoning.
- Avoid unnecessary function nesting; use simple, readable code structure when possible. Only use nested functions if it is natural for the algorithm
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
## Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

## Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```
- Format your skills with:
```skills
skill1, skill2, ...
```
## Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{{{'age': 20, 'city': 'New York'}}}}
```

```skills
'bfs', 'two_pointers'
```

### Output Format:
Your final response **must be valid JSON** exactly matching the schema below. Please output **ONLY** the JSON object. Do not include any extra things.
JSON SCHEMA TO FOLLOW:
{{
    "tasks": [
        {{
            "code": "```python\n<code snippet of the new task1>\n```",
            "input": "```input\n<input of the new task1>\n```",
            "skill": ["<skill1>", "<skill2>", ...]
            "reasoning": "<reasoning of why this task is a good task to address the student's weakness>"
        }},
        {{
            "code": "```python\n<code snippet of the new task2>\n```",
            "input": "```input\n<input of the new task2>\n```",
            "skill": ["<skill1>", "<skill2>", ...]
            "reasoning": "<reasoning of why this task is a good task to address the student's weakness>"
        }},
        ...
    ]
}}

## Representative Performance Profile:
{performance_profile}

## Categorized Algorithmic Skills:
{skills}
""".strip()



code_output_prompt = """
## Task: Create a New Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input

{failure_info}

Skills:
{skill_str}

### Code Requirements:
- The provided skill(s) should be the main testing concept of the code snippet.
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi-step reasoning.
- Avoid unnecessary function nesting; use simple, readable code structure when possible. Only use nested functions if it is natural for the algorithm
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{{{'age': 20, 'city': 'New York'}}}}
```

### Evaluation Criteria:
- Relevance (High Priority), the task should mainly focus on examining the ability of the test subject to understand / reason / apply the skill(s).
- Executability, your code should be executable given your input
- Difficulty in predicting your ```input``` from 1) your ```python``` code and 2) the deterministic ```output``` that will be obtained from your ```input```. Focus on either algorithmic reasoning or logic complexity. For example, you can define complex data structure classes and operate on them like trees, heaps, stacks, queues, graphs, etc, or use complex control flow, dynamic programming, recursions, divide and conquer, greedy, backtracking, etc
- Creativity, the code needs to be sufficiently different from the provided reference snippets
- Restricted usage of certain keywords and packages, you are not allowed to use the following words in any form, even in comments: <|BANNED_KEYWORDS|>

First, carefully devise a clear plan: e.g. how to make the skill(s) the main testing concept of the task, identify how your snippet will be challenging and creative. If there is previous failure information, think about how to resolve the failure and create acceptable code snippet. Then, write the final code snippet and its inputs.
""".strip()

code_output_iterative_prompt = """
## Task: Create a New Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input

{failure_info}

Skills:
{skill_str}

### Previous Synthesis Context:
The code snippet provided below is created one iteration earlier for the skill above, plus current model's performance on this snippet. The performance is defined as the fraction of correct solutions over 10 independent solving attempts on the synthesized task (e.g., 3 correct solutions out of 10 attempts results in 0.3).
Code Snippet:
{prev_code}
Input:
{prev_input}
Performance:
{prev_performance}

You need to take this context into account while creating the new code snippet:
- Adjust difficulty based on previous performance: if high (>0.8), increase complexity and reasoning requirements.
- Ensure the new snippet is distinct from the previous one to maximize diversity.

### Code Requirements:
- The provided skill(s) should be the main testing concept of the code snippet.
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi-step reasoning.
- Avoid unnecessary function nesting; use simple, readable code structure when possible. Only use nested functions if it is natural for the algorithm
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{{{'age': 20, 'city': 'New York'}}}}
```

### Evaluation Criteria:
- Relevance (High Priority), the task should mainly focus on examining the ability of the test subject to understand / reason / apply the skill(s).
- Executability, your code should be executable given your input
- Difficulty in predicting your ```input``` from 1) your ```python``` code and 2) the deterministic ```output``` that will be obtained from your ```input```. Focus on either algorithmic reasoning or logic complexity. For example, you can define complex data structure classes and operate on them like trees, heaps, stacks, queues, graphs, etc, or use complex control flow, dynamic programming, recursions, divide and conquer, greedy, backtracking, etc
- Creativity, the code needs to be sufficiently different from the provided reference snippets
- Restricted usage of certain keywords and packages, you are not allowed to use the following words in any form, even in comments: <|BANNED_KEYWORDS|>

First, carefully devise a clear plan: e.g. how to make the skill(s) the main testing concept of the task, identify how your snippet will be challenging and creative. If there is previous failure information, think about how to resolve the failure and create acceptable code snippet. Then, write the final code snippet and its inputs.
""".strip()

code_output_mutation_prompt = """
## Task: Create Multiple Variants of an Original Task by Systematically Applying Complexity Attributes to Increase Complexity and Reasoning Requirements.

Original task is a code reasoning deduction task that demands deep algorithmic reasoning to deduce the output from the input and Python code snippet.
You will be provided with:
- Original task: one Python code snippet and one input.
- Skill: the algorithmic skill(s) that the original task mainly tests.
- Complexity attributes: a list of complexity attributes that you can consider to increase the complexity and reasoning requirements of the original task. Be aware that not all attributes are applicable to the original task. You need to first determine which attributes could be used.

## Your Mission:

### Step 1: Analyze the Original Task
Examine the provided task and identify:
- Current complexity level for each **applicale** attribute
- Potential for complexity enhancement in each dimension

### Step 2: Generate Diverse Variants
Create **at least 10 different variants** via the following approaches:
- Single-attribute mutations: Enhance only one complexity dimension
- Multi-attribute mutations: Modify multiple but not all attributes at same time
- Comprehensive mutations: Adjust all applicable attributes together

### Requirements for Each Variant:

#### Code Requirements:
 The provided skill(s) should be the main testing concept of the code snippet.
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi step reasoning.
- Avoid unnecessary function nesting; use simple, readable code structure when possible. Only use nested functions if it is natural for the algorithm
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

#### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

#### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{{{'age': 20, 'city': 'New York'}}}}
```

#### Complexity Enhancements:
- Preserve the core algorithmic skill being tested
- Consider the original task's difficulty (pass rate) when increasing complexity. The variants should not be too easy (pass rate  = 100%) or too hard (pass rate  = 0%).
- Each variant should be meaningfully different from others
- Maintain solvability while increasing complexity
- Output the complexity attributes that you considered for each variant and one concise description of how you increase the complexity by each attribute and why it is more complex than the original task.

### Output Format:
Your final response **must be valid JSON** exactly matching the schema below. Please output **ONLY** the JSON object. Do not include any extra things.
JSON SCHEMA TO FOLLOW:
{{
    "variant_1": {{
        "complexity_attributes": [
            <name of the complexity attribute being considered>,
            ...
        ],
        "description": <description of how variant_1 is more complex than the original task by the complexity attributes you considered>,
        "code": "```python\n<code snippet of variant_1>\n```",
        "input": "```input\n<input of variant_1>\n```",
    }},
    "variant_2": ...,
    ...
}}

### Generation Strategy:

1. **Breadth First**: Generate variants that explore different complexity attributes before combining them
2. **Difficulty Progression**: Include variants ranging from moderate increases to significant complexity jumps
3. **Reasoning Diversity**: Create variants that fail for different reasons (timeout, overflow, logic errors, edge cases)

### Evaluation Criteria:
- Relevance to core algorithmic skills. Each variant should test the same core skill.
- Complexity increases should be meaningful, not artificial.
- Variants should explore different failure modes and challenge different aspects of problem-solving.
- Executability, your code should be executable given your input.

Remember: The goal is to create a rich set of practice problems that gradually build expertise by challenging different aspects of the skill through varied complexity enhancements.

## Original Task:
Code:
{code}

Input:
{input}

## Skill:
{skill}

## Complexity Attributes:
{complexity_attributes}
""".strip()

code_output_crossover_prompt = """
## TASK: Create a New Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input and with Novel Skill Combination

### Available Information:

**Skill Pool:**
{skill_pool}
- A comprehensive list of algorithmic coding skills along with their descriptions that you can choose from to create new combinations

**Target Core Skill:**
{target_skill}
- The primary skill that must be the central focus of your code snippet along with its description

**Existing Skill Combinations:**
{existing_combinations}
- Previously created combinations of the target skill with other skills

### Your Task:
You need to create a novel crossover by combining the target skill with other compatible skills from the skill pool. This requires:
1. Examining existing combinations to understand what has already been done
2. Identifying skills from the pool that would naturally work with the target skill but haven't been combined yet
3. Creating a code snippet where all chosen skills are essential and interconnected to solve the problem

The key challenge is to ensure your skill combination is both NEW (not in existing combinations) and MEANINGFUL (skills genuinely complement each other rather than being artificially forced together).

### Code Requirements:
- The target skill MUST be the central concept, with other skills naturally supporting or enhancing it
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- The snippet should demonstrate clear interdependence between all skills in the combination (not just using skills in isolation)
- Require state tracking across multiple data transformations, ensuring multi-step reasoning
- Avoid unnecessary function nesting; use simple, readable code structure when possible
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`
{remove_input_from_snippet_prompt}{remove_after_return_prompt}

### Skill Combination Requirements:
- Your new combination must include the target skill plus at least one other skill from the skill pool
- The combination should be meaningfully different from all existing combinations
- Skills should work together synergistically, not just appear sequentially
- Justify why your chosen skills complement each other naturally and how they work together in your code snippet

### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{{{'age': 20, 'city': 'New York'}}}}
```

### Output Format:
Your final response **must be valid JSON** exactly matching the schema below. Please output **ONLY** the JSON object. Do not include any extra things.
JSON SCHEMA TO FOLLOW:
{{
    "skill_combination": [<target_skill>, <new_combined_skill_1>, <new_combined_skill_2>, ...],
    "crossover_description": <brief explanation of why and how these skills work well together>,
    "code": "```python\n<code snippet of the new skill combination>\n```",
    "input": "```input\n<input of the new skill combination>\n```",
}}

### Evaluation Criteria:
- Novelty (Critical): The skill combination must be different from all existing combinations
- Integration (High Priority): All skills must work together meaningfully, not just appear independently
- Target Skill Focus: The target skill should be the dominant concept being tested
- Executability: Your code should run successfully with the provided input
- Difficulty: The task should require understanding the interaction between skills, not just individual skill mastery
- Creativity: The scenario should be distinct from existing code snippets
- Complexity: Focus on algorithmic reasoning or logic complexity (e.g., complex data structures, control flow, dynamic programming, recursion)
- Restricted Keywords: You cannot use the following words in any form: <|BANNED_KEYWORDS|>

### Process:
- First, analyze the existing combinations to understand patterns and identify gaps
- Second, propose your new skill combination with clear justification
- Third, devise a plan for how these skills will interact in your code
- Fourth, implement the code snippet ensuring all skills are essential to the solution and following the code requirements
- Finally, create an input that exercises the full combination

If there is previous failure information, address those issues explicitly in your new attempt.
""".strip()


code_output_curriculum_prompt = """
You are a senior mentor specializing in algorithmic programming education.

## Task
Given a student's performance profile on code deduction tasks, create **five** new Python code snippets, each with a matching test input, that target the student's identified weaknesses. The three tasks should address different weaknesses or explore the same weakness from different angles.

**Code Deduction Task Definition**: The student is given a code snippet and an input, then must deduce the correct output through reasoning (without executing the code).

## Provided Information
1. Student Performance Profile
- Contains tasks across three difficulty levels: **easy**, **medium**, and **hard**.
- Each level includes 10 example tasks, each annotated with:
  - **Student score**: Average accuracy across 10 attempts (range: 0.0 to 1.0).
  - **Tested skills**: The algorithmic concepts required to solve the task.
  - **Category**: The broader topic area of the task.
- Tasks are randomly sampled from the student's full performance history.
2. Categorized Algorithmic Skills
- A taxonomy of algorithmic skills, each with a description.
- The student should progressively master these skills through targeted practice.

## Your Objectives
1. **Analyze** the performance profile to identify the student's weaknesses.
2. **Create five** new code snippets, each with one matching test input, that address the identified weaknesses.
3. **Specify** the primary skills tested by each new task (must be drawn from the provided skill taxonomy).
4. **Reason** why each new task is a good task to address the student's weakness (concisely).

## Analysis Guidelines
- Examine each example task carefully to identify patterns in the student's reasoning deficiencies.
- Prioritize skills where the student shows **consistent difficulty**, especially across "medium" and "hard" tasks.
- Favor weaknesses that appear across **multiple tasks or difficulty levels** over isolated failures.
- When uncertain, prioritize **algorithmic reasoning and complexity gaps** over minor accuracy issues.
- Select **atomic skills** (i.e., specific, indivisible concepts) from the skill taxonomy.
- Ensure **novelty**: the new task should differ meaningfully from the sampled examples.
- Ensure **diversity**: the five tasks should vary in structure, skills tested, or approach.
- Ensure **difficulty**: the new task's difficulty should be appropriate and challenging enough to address weaknesses, but achievable with effort.

## Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi-step reasoning.
- Avoid unnecessary function nesting; use simple, readable code structure when possible. Only use nested functions if it is natural for the algorithm
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
## Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

## Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```
- Format your skills with:
```skills
skill1, skill2, ...
```
## Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{{{'age': 20, 'city': 'New York'}}}}
```

```skills
'bfs', 'two_pointers'
```

### Output Format:
Your final response **must be valid JSON** exactly matching the schema below. Please output **ONLY** the JSON object. Do not include any extra things.
JSON SCHEMA TO FOLLOW:
{{
    "tasks": [
        {{
            "code": "```python\n<code snippet of the new task1>\n```",
            "input": "```input\n<input of the new task1>\n```",
            "skill": ["<skill1>", "<skill2>", ...]
            "reasoning": "<reasoning of why this task is a good task to address the student's weakness>"
        }},
        {{
            "code": "```python\n<code snippet of the new task2>\n```",
            "input": "```input\n<input of the new task2>\n```",
            "skill": ["<skill1>", "<skill2>", ...]
            "reasoning": "<reasoning of why this task is a good task to address the student's weakness>"
        }},
        ...
    ]
}}

## Representative Performance Profile:
{performance_profile}

## Categorized Algorithmic Skills:
{skills}
""".strip()

code_function_prompt_old = """
## Task: Output {num_inputs} Inputs that can be plugged into the following Code Snippet to produce diverse Outputs, and give a message related to the given snippet.

Using the code snippet provided below, design {num_inputs} inputs that can be plugged into the code snippet to produce a diverse set of outputs. A subset of your given input and its deterministically produced outputs will be given to a test subject to deduce the function, which is meant to be an I.Q. test. You can also leave a message to the test subject to help them deduce the code snippet.

### Input Requirements:
- Provide {num_inputs} valid inputs for the code snippet
- For each input, format multiple arguments with commas between them
- Remember to add quotes around string arguments
- Each input should be individually wrapped in ```input``` tags

### Message Requirements:
- Leave a message to the test subject to help them deduce the code snippet
- The message should avoid **explicitly referencing** the testing skill(s).
- The message should be wrapped in ```message``` tags
- The message can be in any form, can even be formed into a coding question, or a natural language instruction what the code snippet does
- You cannot provide the code snippet in the message

### Formatting:
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```input
'John', {{'age': 20, 'city': 'New York'}}
```
```input
'Sammy', {{'age': 37, 'city': 'Los Angeles'}}
```

### Evaluation Criteria:
- Executability, your code should be executable given your inputs
- Coverage, the inputs and outputs should cover the whole input space of the code snippet, able to deduce the code snippet from the inputs and outputs
- Creativity, the inputs need to be sufficiently different from each other
- The overall selection of inputs and message combined should be challenging for the test subject, but not impossible for them to solve
- The message should implicitly encourage the test subject to use the testing skill(s) to solve the task.
First, carefully devise a clear plan: e.g. how to make the skill(s) the main testing concept of the task, understand the code snippet, then identify how your proposed inputs have high coverage, and why the inputs will be challenging and creative. Then, write the inputs and message. Remember to wrap your inputs in ```input``` tags, and your message in ```message``` tags.

### Code Snippet:
```python
{code}
```
"""

# code_function_prompt = """
# ## Task: Generate a natural coding problem related to the code snippet, and {num_inputs} inputs that can be plugged into the code snippet to produce a diverse set of outputs.

# Using the code snippet provided below, design comprehensive {num_inputs} inputs that can be plugged into the code snippet to produce a diverse set of outputs. A subset of your given input and its deterministically produced outputs will be given to a test subject to deduce the function, which is meant to be an I.Q. test. You should also create a natural coding problem for which the given code snippet would be a valid solution, and your generated inputs would be the test inputs for the problem.

# ### Input Requirements:
# - Provide {num_inputs} valid inputs for the code snippet that comprehensively cover the code's behavior
# - For each input, format multiple arguments with commas between them
# - Remember to add quotes around string arguments
# - Each input should be individually wrapped in ```input``` tags
# - Ensure diversity: inputs should test different aspects and branches of the code 

# ### Problem Requirements:
# - Create a natural coding problem statement that describes what needs to be solved
# - The provided code snippet must be a correct and complete solution to the problem you describe
# - Ensure that solving the problem statement would naturally lead to implementing logic similar to the given code snippet
# - The problem statement should be wrapped in ```problem``` tags
# - You cannot provide the code snippet in the problem statement

# ### Formatting:
# - Format your input with:
# ```input
# arg1, arg2, ...
# ```

# ### Example Format:
# ```input
# 'John', {{'age': 20, 'city': 'New York'}}
# ```
# ```input
# 'Sammy', {{'age': 37, 'city': 'Los Angeles'}}
# ```

# ### Evaluation Criteria:
# - Executability, the code should be executable given your inputs
# - Coverage, the inputs should cover the whole input space of the code snippet
# - Creativity, the inputs need to be sufficiently different from each other
# - The overall selection of inputs and message combined should be challenging for the test subject, but not impossible for them to solve
# First, carefully devise a clear plan: e.g. understand the code snippet, then identify how your proposed inputs have high coverage, and why the inputs will be challenging and creative. Then, write the inputs and message. Remember to wrap your inputs in ```input``` tags, and your message in ```message``` tags.

# ### Code Snippet:
# ```python
# {code}
# ```
# """


code_function_prompt = """
## Task: Generate a natural coding problem related to the code snippet, and {num_inputs} inputs that can be plugged into the code snippet to produce a diverse set of outputs.

Using the code snippet provided below, design comprehensive {num_inputs} inputs that can be plugged into the code snippet to produce a diverse set of outputs. A subset of your given input and its deterministically produced outputs will be given to a test subject to deduce the function, which is meant to be an I.Q. test. You should also create a natural coding problem for which the given code snippet would be a valid solution, and your generated inputs would be the test inputs for the problem.

### Input Requirements:
- Provide {num_inputs} valid inputs for the code snippet that comprehensively cover the code's behavior
- For each input, format multiple arguments with commas between them
- Remember to add quotes around string arguments
- Each input should be individually wrapped in ```input``` tags
- Ensure diversity: inputs should test different aspects and branches of the code 

### Problem Requirements:
- Create a natural coding problem that clearly describes what needs to be solved. Do not include examples or constraints.
- Write in an engaging, scenario-based style when possible (e.g., "You are given an array of meeting times..." or "A company needs to process customer orders...")
- The provided code snippet must be a correct and complete solution to the problem you describe
- Ensure that solving the problem statement would naturally lead to implementing logic similar to the given code snippet
- The problem statement should be wrapped in ```problem``` tags
- You cannot include or leak the code snippet in the problem statement

### Formatting:
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```input
'John', {{'age': 20, 'city': 'New York'}}
```
```input
'Sammy', {{'age': 37, 'city': 'Los Angeles'}}
```

### Evaluation Criteria:
- Executability, the code should be executable given your inputs
- Coverage, the inputs should cover the whole input space of the code snippet
- Creativity, the inputs need to be sufficiently different from each other
- The overall selection of inputs and message combined should be challenging for the test subject, but not impossible for them to solve
- Problem Quality, The problem statement should read like a real coding assessment question
First, carefully devise a clear plan: e.g. understand the code snippet, then identify how your proposed inputs have high coverage, and why the inputs will be challenging and creative. Then, write the inputs and message. Remember to wrap your inputs in ```input``` tags, and your message in ```message``` tags.

### Code Snippet:
```python
{code}
```
"""

code_function_hint_prompt = """
## Task: Generate progressive hints for a coding problem

The coding problem below was generated based on a code snippet and has proven too challenging for test subjects to solve. Your task is to create a series of progressive hints that guide the solver toward the solution WITHOUT giving away the complete answer

### Original Problem:
```problem
{problem}
```

### Code Snippet:
```python
{code}
```

### Hints Requirements:
- Generate 3-4 progressive hints that gradually reveal the solution approach (code snippet)
- Follow the hint style:
    * Start with high-level intuition or pattern recognition
    * Progress to algorithm/data structure suggestions
    * Then provide implementation details or key insights
    * Final hint can outline the approach in a coarse level but should not reveal the actual code implementation
- Each hint should build upon the previous one
- Hints should be concise, to the point, and guide thinking without revealing the exact solution
- Ensure the hints are grounded to the code snippet, which is the solution to the problem
- Hints should be wrapped in ```hint``` tags

### Formatting:
- Format your hints with:
```hint
<hint_content>
```

### Example Format:
```hint
Can you think of this problem in terms of a decision tree, where at each step, we have n decisions, where n is the size of the array?
```
```hint
We can use backtracking to recursively traverse these paths and make decisions to choose an element at each step.
```

### Evaluation Criteria:
- Progressive Difficulty: Each hint should reveal slightly more than the previous
- Guidance Quality: Hints should genuinely help stuck test subjects to make progress, not just give answers
- Clarity: Use simple language and clear explanations
First, carefully devise a clear plan: e.g. understand the code snippet, then identify what concepts or patterns might not be obvious, what edge cases could trip up solvers, what algorithmic insight is needed?. Then, create the hints. Remember to wrap your hints in ```hint``` tags.
"""



CODE_IO_FAILURE_TEMPLATES: Dict[str, str] = {
    "default": """
Design a new and unique Python code snippet that 
    1. **MAINLY FOCUSED ON EXAMINING** the ability of the test subject to understand / reason / apply the provided skill(s).
    2. demands deep algorithmic reasoning to {task_requirement}.
    """.strip(),
    "failure": """
The code snippet provided below is a previous attempt at proposing a {task_label}. It is **rejected** for the reason shown below.

### Failure Code Snippet:
```python
{code}
```

### Reason for rejection: {reason}
{feedback}

You should deeply analyze and reflect on the rejection. Then design a new and unique Python code snippet or modify the existing code snippet so that it 
    1. specifically resolves the reason for rejection.
    2. **MAINLY FOCUSED ON EXAMINING** the ability of the test subject to understand / reason / apply the provided skill(s).
    3. demands deep algorithmic reasoning to {task_requirement}.
    """.strip()
}

TASK_FAILURE_PROMPTS: Dict[str, Dict[str, Any]] = {
    "code_out": {
        "label": "Python code snippet with one matching input",
        "requirement": "deduce the output from the input",
        "default": CODE_IO_FAILURE_TEMPLATES["default"],
        "failure": CODE_IO_FAILURE_TEMPLATES["failure"],
    },
    "code_in": {
        "label": "Python code snippet with one matching input",
        "requirement": "deduce one possible input from a given output.",
        "default": CODE_IO_FAILURE_TEMPLATES["default"],
        "failure": CODE_IO_FAILURE_TEMPLATES["failure"],
    },
    "code_func": {
        "label": "code-function reconstruction task",
        "requirement": "reconstruct the underlying function from observed behaviours",
        "default": CODE_IO_FAILURE_TEMPLATES["default"],
        "failure": CODE_IO_FAILURE_TEMPLATES["failure"],
    },
}

code_input_predictor_prompt = """
# Task: Provide One Possible Input of a Python Code Snippet Given the Code and Output
Given the following Code Snippet and the Output, think step by step then provide one possible input that produced the output. The input needs to be wrapped in ```input``` tags. Remember if an argument is a string, wrap it in quotes. If the function requires multiple arguments, separate them with commas.

# Code Snippet:
```python
{snippet}
```

# Output:
```output
{output}
```

# Output Format:
```input
arg1, arg2, ...
```
# Example Output:
```input
'John', {{'age': 20, 'city': 'New York'}}
```
"""

code_output_predictor_prompt = """
# Task: Deduce the Output of a Python Code Snippet Given the Code and Input
Given the following Code Snippet and the Input, think step by step then deduce the output that will be produced from plugging the Input into the Code Snippet. Put your output in ```output``` tags. Remember if the output is a string, wrap it in quotes. If the function returns multiple values, remember to use a tuple to wrap them.

# Code Snippet:
```python
{snippet}
```

# Input:
```input
{input_args}
```

# Example Output:
```output
{{'age': 20, 'city': 'New York'}}
```
"""

code_suffix = "\nf(<|YOUR INPUT WILL BE PLUGGED HERE|>)"

code_function_predictor_prompt = """
# Task: Deduce the Function that Produced the Outputs from the Inputs
Given a set of input/output pairs and a problem that describes the function, think through the problem step by step to deduce a general code snippet. This code should produce the hidden outputs from the hidden inputs, matching the original data-generating code that created the input/output pairs. Place your final answer inside python tags! It may be helpful to work through each input/output pair individually to test your function. If your function doesn't work as expected, revise it until it does. The final code snippet will be used to evaluate your response, which is wrapped in ```python``` tags.

# Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f()`, anything after will be removed

# Input and Output Pairs:
{input_output_pairs}

# Problem:
```problem
{problem}
```

# Example Output:
```python
def f(a):
    return a
```

Name your entry function `f()`!!!
"""

remove_input_from_snippet_prompt = "- Do not have the test input anywhere in the code snippet, provide it in the input section."

remove_singleton_variables_prompt = "- All variable declarations must be inside the main function `f` or within functions `f` make calls to. Any variables declared outside of functions will be removed.\n"

instruction_following = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}\nAssistant: <think>"

instruction_following_system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

def generate_task_prompt(
    task_type: str,
    tasks: Dict[str, Any],
    banned_keywords: List[str],
    remove_after_return: bool = False,
    num_inputs: int = 10,
    remove_input_from_snippet: bool = False,
    reject_info: Optional[Dict[str, Any]] = None,
    complexity_attributes: Optional[List[str]] = None, # For mutation
    existing_combinations: Optional[Dict[Tuple, Any]] = None, # For crossover
    skill_pool: Optional[List[str]] = None, # For crossover
    cluster_skills: Optional[List[Dict[str, Any]]] = None, # For curriculum
    mutate: bool = False,
    crossover: bool = False,
    curriculum: bool = False,
    prev_code: Optional[str] = None,
    prev_input: Optional[str] = None,
    prev_performance: Optional[float] = None,
):
    if not curriculum:
        skill_str = f"{tasks['skill']}: {tasks['skill_description']}"
    # skill_str = tasks["skill"]

    if task_type == "code_in":
        if curriculum:
            examples = sample_examples(tasks, task_type)
            performance_profile = _format_performance_profile(examples, task_type)
            cluster_skills_str = _format_skills_prompt(cluster_skills)
            return code_input_curriculum_prompt.format(
                performance_profile=performance_profile,
                skills=cluster_skills_str,
                remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
                remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
            ).replace(
                '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
            )
        if mutate:
            task_code = tasks["code"]
            task_input = tasks["input"]
            complexity_attributes = [f"{attr}:{description}" for attribute in complexity_attributes for attr, description in attribute.items()]
            return code_input_mutation_prompt.format(
                remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
                remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
                code=task_code,
                input=task_input,
                skill=skill_str,
                complexity_attributes="\n".join(complexity_attributes),
            ).replace(
                '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
            )

        failure_info = _make_failure_info(reject_info, task_type=task_type)

        if crossover:
            existing_combinations_str = "".join(
                f"{list(comb)}\n"
                for comb, code in existing_combinations.items()
            )

            return code_input_crossover_prompt.format(
                failure_info=failure_info,
                target_skill=skill_str,
                skill_pool="\n".join(skill_pool),
                existing_combinations=existing_combinations_str,
                remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
                remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
            ).replace(
                '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
            )

        if prev_code and prev_input and prev_performance:
            return code_input_iterative_prompt.format(
                failure_info=failure_info,
                skill_str=skill_str,
                remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
                remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
                prev_code=prev_code,
                prev_input=prev_input,
                prev_performance=prev_performance,
            ).replace(
                '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
            )

        return code_input_prompt.format(
            failure_info=failure_info,
            skill_str=skill_str,
            remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
            remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
        ).replace(
            '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
        )

    if task_type == "code_out":
        if curriculum:
            examples = sample_examples(tasks, task_type)
            performance_profile = _format_performance_profile(examples, task_type)
            cluster_skills_str = _format_skills_prompt(cluster_skills)
            return code_output_curriculum_prompt.format(
                performance_profile=performance_profile,
                skills=cluster_skills_str,
                remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
                remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
            ).replace(
                '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
            )
        if mutate:
            task_code = tasks["code"]
            task_input = tasks["input"]
            complexity_attributes = [f"{attr}:{description}" for attribute in complexity_attributes for attr, description in attribute.items()]
            return code_output_mutation_prompt.format(
                remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
                remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
                code=task_code,
                input=task_input,
                skill=skill_str,
                complexity_attributes="\n".join(complexity_attributes),
            ).replace(
                '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
            )

        failure_info = _make_failure_info(reject_info, task_type=task_type)

        if crossover:
            existing_combinations_str = "".join(
                f"{list(comb)}\n"
                for comb, code in existing_combinations.items()
            )

            return code_output_crossover_prompt.format(
                failure_info=failure_info,
                target_skill=skill_str,
                skill_pool="\n".join(skill_pool),
                existing_combinations=existing_combinations_str,
                remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
                remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
            ).replace(
                '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
            )
        
        if prev_code and prev_input and prev_performance:
            return code_output_iterative_prompt.format(
                failure_info=failure_info,
                skill_str=skill_str,
                remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
                remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
                prev_code=prev_code,
                prev_input=prev_input,
                prev_performance=prev_performance,
            ).replace(
                '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
            )

        return code_output_prompt.format(
            failure_info=failure_info,
            skill_str=skill_str,
            remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
            remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else ''),
        ).replace(
            '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
        )

    if task_type == "code_func":
        code = tasks.get("code", "")
        if reject_info:
            return code_function_hint_prompt.format(
                problem=tasks["problem"],
                code=code,
            )
        return code_function_prompt.format(
            num_inputs=num_inputs,
            code=code,
        )
    

def generate_reflection_prompt(skill_list: List[str], code_snippet: str):
    return skill_reflection_prompt.format(
        skill_list=skill_list,
        code_snippet=code_snippet
    )


def get_code_problem_predictor_prompt(problem_type: str, snippet: str, input: str = None, output: str = None, problem: str = None, input_output_pairs: List[Tuple[str, str]] = None) -> str:
    if problem_type.endswith("code_in"):
        return code_input_predictor_prompt.format(snippet=snippet, output=output)
    elif problem_type.endswith("code_out"):
        return code_output_predictor_prompt.format(snippet=snippet, input_args=input)
    elif problem_type.endswith("code_func"):
        input_output_pairs_string = ""
        for i, (input, output) in enumerate(input_output_pairs):
            input_output_pairs_string += f"```input_{i}\n{input}\n```\n```output_{i}\n{output}\n```\n"
        return code_function_predictor_prompt.format(input_output_pairs=input_output_pairs_string, problem=problem)
    else:
        raise ValueError(f"Invalid problem type: {problem_type}")

def _get_task_failure_prompts(task_type: str) -> Dict[str, Any]:
    try:
        return TASK_FAILURE_PROMPTS[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task type for failure prompt: {task_type}") from exc


def _format_failure_prompt(template: str, *, task_label: str, task_requirement: str, reject_info: Optional[Dict[str, Any]]) -> str:
    info = reject_info or {}
    # Normalise fields with sensible fallbacks to avoid KeyErrors.
    values = {
        "task_label": task_label,
        "task_requirement": task_requirement,
        "code": info.get("code", ""),
        "reason": info.get("reason", ""),
        "feedback": info.get("feedback", ""),
    }
    return template.format(**values).strip()


def _select_failure_template(task_prompts: Dict[str, Any], reject_info: Optional[Dict[str, Any]]) -> str:
    if not reject_info:
        return task_prompts["default"]
    return task_prompts.get("failure", task_prompts["default"])

def _make_failure_info(reject_info: Dict[str, Any] | None, task_type: str | None = None) -> str:
    task_prompts = _get_task_failure_prompts(task_type)
    template = _select_failure_template(task_prompts, reject_info)
    return _format_failure_prompt(
        template,
        task_label=task_prompts["label"],
        task_requirement=task_prompts["requirement"],
        reject_info=reject_info,
    )

def _format_performance_profile(examples: Dict[str, List[Dict[str, Any]]], task_type: str) -> str:
    performance_profile = ""
    if task_type == "code_in":
        param = "Output"
    elif task_type == "code_out":
        param = "Input"
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    for level, items in examples.items():
        performance_profile += f"### **{level.capitalize()}** Difficulty Level:\n"
        for idx, item in enumerate(items):
            performance_profile += f"{idx+1}. Category: {item['category']} | Skill: {item['skill']} | Student Score: {item['difficulty']}\n"
            performance_profile += f"   - {param}: {item[param.lower()]}\n"
            performance_profile += f"   - Code Snippet: \n{item['code'].strip()}\n\n"
    return performance_profile

def _format_skills_prompt(skills: List[Dict[str, Any]]) -> str:
    formatted_skills = ""
    for category in skills:
        category_name = category["category"]
        formatted_skills += f"### {category_name}:\n"
        for skill in category["members"]:
            formatted_skills += f"   - {skill['skill']}: {skill['description']}\n"

    return formatted_skills