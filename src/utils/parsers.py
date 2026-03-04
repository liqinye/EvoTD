import re
import ast
from typing import Tuple, Dict, List, Any, Optional
import json


def strip_think_token(response: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    clean = re.sub(r'\n\s*\n+', '\n\n', clean)
    return clean.strip()

def strip_json(response: str) -> str:
    clean = re.sub(r"```json\s*|\s*```", "", response)
    return clean.strip()

def has_test_input(snippet_code: str) -> bool:
    test_patterns = [
        r"(?i)#\s*(test|example)",  # Match any test/example comment
        r"\b(input|test_input|sample_input)\b\s*=",  # Common test variable names
        r"\b\w*input\w*\s*=\s*",    # Match any variable containing "input"
        r"\b(expected|output|result)\s*=\s*",
        r"\bassert\b",
        r"print\s*\(\s*f\(",
        r"f\(\[.*\]\)",
        r"f\([^)]*\)\s*(#|$)",
        r"^\s*input\s*$",  # Match lines containing only "input"
    ]

    return any(
        re.search(pattern, snippet_code, re.MULTILINE)
        for pattern in test_patterns
    )

def parse_imports(code_snippet: str) -> List[str]:
    imports = []
    try:
        tree = ast.parse(code_snippet)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Reconstruct import line from AST node
                if isinstance(node, ast.Import):
                    import_line = "import " + ", ".join(
                        [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                            for alias in node.names]
                    )
                else:
                    module = node.module or ""
                    import_line = f"from {module} import " + ", ".join(
                        [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                            for alias in node.names]
                    )
                    if node.level > 0:
                        import_line = f"from {'.' * node.level}{module} import " + ", ".join(
                            [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                                for alias in node.names]
                        )
                imports.append(import_line)
    except Exception as e:
        import_pattern = r"^\s*(?:from|import)\s+.*$"
        imports = [i.strip() for i in re.findall(import_pattern, code_snippet, re.MULTILINE)]
    return imports

def parse_error(error_message: str) -> str:
    # split by colon
    error_message = error_message.split(':')[0]
    return error_message.strip()

def remove_comments_and_docstrings(code: str) -> str:
    """
    Remove all comments and docstrings from the code.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
                # Remove all leading docstrings
                while node.body and isinstance(node.body[0], ast.Expr):
                    expr = node.body[0].value
                    if isinstance(expr, (ast.Str, ast.Constant)) and (
                        isinstance(expr.value, str) if isinstance(expr, ast.Constant) else True
                    ):
                        node.body.pop(0)
                    else:
                        break
        
        # Convert back to code - AST unparse already removes comments
        code_without_docstrings = ast.unparse(tree)
        
        # Only remove empty lines and trim whitespace
        lines = [
            line.rstrip()
            for line in code_without_docstrings.split('\n')
            if line.strip()
        ]
        
        return '\n'.join(lines)
    except Exception as e:
        return code  # Return original code if parsing fails


def remove_any_not_definition_imports(code: str) -> str:
    """
    Remove anything that is not a definition or import.
    Preserves: 
    - Import/From imports
    - Class definitions
    - Function/AsyncFunction definitions
    Removes:
    - Top-level assignments
    - Standalone expressions
    - Constant declarations
    """
    class DefinitionFilter(ast.NodeTransformer):
        def visit_Module(self, node):
            # Keep only definitions and imports (explicitly exclude assignments)
            node.body = [
                n for n in node.body
                if isinstance(n, (
                    ast.Import,
                    ast.ImportFrom,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef
                ))
            ]
            return node

    try:
        tree = ast.parse(code)
        tree = DefinitionFilter().visit(tree)
        ast.fix_missing_locations(tree)

        # Remove empty lines and format
        cleaned = ast.unparse(tree)
        return '\n'.join([line for line in cleaned.split('\n') if line.strip()])

    except Exception as e:
        return code

def rename_constructor(code: str) -> str:
    """
    For each class in the code, rename its first method (regardless of name) to __init__.
    Only rename if the method is inside the class definition.
    """
    # Pattern to match class blocks (class ...: ...), capturing the class name and body
    class_block_pattern = re.compile(
        r'(class\s+\w+\s*:\s*\n)((?:[ \t]+.+\n?)*)', re.MULTILINE
    )

    def class_block_replacer(match):
        class_header = match.group(1)
        body = match.group(2)
        # Find the first method definition in the class body
        method_pattern = re.compile(r'^([ \t]+)def\s+(\w+)\s*\(', re.MULTILINE)
        method_match = method_pattern.search(body)
        if method_match:
            indent = method_match.group(1)
            old_method = method_match.group(2)
            # Replace only the first occurrence
            body = method_pattern.sub(
                rf'\1def __init__(', body, count=1
            )
        return f'{class_header}{body}'

    code = class_block_pattern.sub(class_block_replacer, code)
    return code

class PrintRemover(ast.NodeTransformer):
    def visit_Expr(self, node):
        # Handle top-level print statements
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
            return None
        return node

    def visit_Call(self, node):
        # Handle print calls in other contexts (like assignments)
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            return ast.Constant(value=None)
        return node

    def _handle_block(self, node):
        self.generic_visit(node)
        if not node.body:
            node.body.append(ast.Pass())
        return node

    def visit_For(self, node):
        return self._handle_block(node)

    def visit_While(self, node):
        return self._handle_block(node)

    def visit_FunctionDef(self, node):
        return self._handle_block(node)

    def visit_AsyncFunctionDef(self, node):
        return self._handle_block(node)

    def visit_If(self, node):
        return self._handle_block(node)

    def visit_With(self, node):
        return self._handle_block(node)

    def visit_Try(self, node):
        self.generic_visit(node)
        
        # Handle main try body
        if not node.body:
            node.body.append(ast.Pass())
            
        # Handle except handlers
        for handler in node.handlers:
            if not handler.body:
                handler.body.append(ast.Pass())
                
        # Handle else clause
        if node.orelse and not node.orelse:
            node.orelse.append(ast.Pass())
            
        # Handle finally clause
        if node.finalbody and not node.finalbody:
            node.finalbody.append(ast.Pass())
            
        return node


def remove_print_statements(code: str) -> str:
    """
    Remove all print statements from the code.
    """
    tree = ast.parse(code)
    tree = PrintRemover().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def parse_code_input_output(
    input_str: str,
    parse_input: bool = True,
    parse_output: bool = True,
    remove_after_return: bool = False,
    remove_comments: bool = False,
    remove_print: bool = False,
    reject_multiple_functions: bool = True,
    reject_test_input_in_code: bool = False,
    f_replace_location: str = 'not_first',
) -> Tuple[bool, Dict[str, str]]:
    """
    Parse the input and output of a code snippet.

    Args:
        input_str: A string containing the code snippet
        parse_input: Whether to parse the input
        parse_output: Whether to parse the output
    """
    # Improved regex patterns with better whitespace handling and optional language specifiers
    code_pattern = r"```(?:python\s*)?\n?(.*?)\n?```"
    input_pattern = r"```input\s*\n?(.*?)\n?```"
    output_pattern = r"```output\s*\n?(.*?)\n?```"

    # Use flags for case-insensitive matching and dotall
    flags = re.DOTALL | re.IGNORECASE
    code_match = re.search(code_pattern, input_str, flags)

    # Check required blocks
    if parse_input:
        input_match = re.search(input_pattern, input_str, flags)
        if not input_match:
            # Try alternative pattern without explicit input block
            input_match = re.search(r"# Input:\s*(.*?)(?=\n```|$)", input_str, flags)
    if parse_output:
        output_match = re.search(output_pattern, input_str, flags)
        if not output_match:
            # Try alternative pattern without explicit output block
            output_match = re.search(r"# Output:\s*(.*?)(?=\n```|$)", input_str, flags)

    # Validate required components
    if not code_match or (parse_input and not input_match) or (parse_output and not output_match):
        print(1)
        return False, {}

    # Extract and clean components
    code_snippet = code_match.group(1).strip()
    input_snippet = input_match.group(1).strip() if parse_input else ""
    output_snippet = output_match.group(1).strip() if parse_output else ""

    # Enhanced function detection and validation
    function_defs = re.findall(r"^\s*def\s+(\w+)\s*\(", code_snippet, re.MULTILINE)
    if not function_defs:
        print(2)
        return False, {}

    if reject_multiple_functions and len(function_defs) > 1:
        print(3)
        return False, {}  # Reject multiple function definitions

    if reject_test_input_in_code and has_test_input(code_snippet):  
        print(4)
        return False, {}

    # Standardize function name to 'f'
    if f_replace_location == 'not_first':
        original_name = function_defs[0]
    elif f_replace_location == 'any_last':
        original_name = function_defs[-1] if 'f' not in function_defs else 'f'
    elif f_replace_location == 'any_first':
        original_name = function_defs[0] if 'f' not in function_defs else 'f'
    elif f_replace_location == 'not_last':
        original_name = function_defs[-1]
    else:
        raise ValueError(f'Invalid f_replace_location: {f_replace_location}')
    if original_name != 'f':
        code_snippet = re.sub(
            rf"def\s+{re.escape(original_name)}\s*\(", 
            "def f(", 
            code_snippet, 
            count=0
        )
        # Replace all calls to the function as well (for recursive functions)
        code_snippet = re.sub(
            rf"\b{re.escape(original_name)}\s*\(",
            "f(",
            code_snippet
        )

    imports: List[str] = parse_imports(code_snippet)

    # before_remove_comments = code_snippet
    # remove comments and docstrings
    if remove_comments:
        code_snippet = remove_comments_and_docstrings(code_snippet)

    # remove anything after return
    if remove_after_return:
        code_snippet = remove_any_not_definition_imports(code_snippet)
    
    # remove print statements
    if remove_print:
        code_snippet = remove_print_statements(code_snippet)

    code_snippet = rename_constructor(code_snippet)
    
    return True, {"code": code_snippet, "input": input_snippet, "output": output_snippet, "imports": imports}


def parse_inputs_message(
    input_str: str,
    num_inputs: int
) -> Tuple[bool, Dict[str, Any]]:
    """
    Parse the last num_inputs inputs and problem statement from a string.

    Args:
        input_str: A string containing the inputs and problem statement
        num_inputs: Number of most recent inputs to parse
    
    Returns:
        A tuple of (success, dict) where dict contains:
        - inputs: List of last num_inputs input strings
        - problem: The problem statement string
        Returns (False, {}) if there aren't enough inputs or problem statement is missing
    """
    # Improved regex patterns with better whitespace handling and optional language specifiers
    input_pattern = r"```input\s*\n?(.*?)\n?```"
    problem_pattern = r"```problem\s*\n?(.*?)\n?```"

    # Use flags for case-insensitive matching and dotall
    flags = re.DOTALL | re.IGNORECASE

    # Check required blocks
    input_matches = re.finditer(input_pattern, input_str, flags)
    if not input_matches:
        # Try alternative pattern without explicit input block
        input_matches = re.finditer(r"# Input:\s*(.*?)(?=\n```|$)", input_str, flags)

    # Get all inputs and take the last num_inputs
    inputs = [match.group(1).strip() for match in input_matches]
    
    # Return early if not enough inputs
    if len(inputs) < num_inputs:
        return False, {}
        
    inputs = inputs[-num_inputs:]  # Take last num_inputs

    problem_match = re.search(problem_pattern, input_str, flags)

    # Try parsing problem statement between <problem> </problem> tags if previous methods failed
    if not problem_match:
        problem_match = re.search(r"<problem>\s*(.*?)\s*</problem>", input_str, flags)

    if not problem_match:
        # Try alternative pattern without explicit problem statement block
        problem_match = re.search(r"# Problem:\s*(.*?)(?=\n```|$)", input_str, flags)

    # Return early if problem statement not found
    if not problem_match:
        return False, {}

    # Extract and clean problem statement
    problem = problem_match.group(1).strip()

    return True, {"inputs": inputs, "problem": problem}


def parse_hint_message(
    input_str: str,
    min_hints: int = 3,
    max_hints: int = 4,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Parse progressive hints wrapped in ```hint``` blocks.

    Args:
        input_str: A string containing hint blocks and optional context
        min_hints: Minimum number of hints required for success
        max_hints: Maximum number of hints allowed

    Returns:
        A tuple of (success, dict) where dict contains:
        - hints: Ordered list of hint strings
        Returns (False, {}) if there aren't enough hints or format is invalid
    """

    flags = re.DOTALL | re.IGNORECASE
    hint_pattern = r"```hint\s*\n?(.*?)\n?```"

    hint_matches = re.findall(hint_pattern, input_str, flags)

    if not hint_matches:
        alternative_pattern = r"#\s*Hint:?\s*(.*?)(?=\n#\s*Hint:?|$)"
        hint_matches = re.findall(alternative_pattern, input_str, flags)

    hints = [hint.strip() for hint in hint_matches if hint.strip()]

    if len(hints) < min_hints:
        return False, {}

    if len(hints) > max_hints:
        hints = hints[:max_hints]

    return True, {"hints": hints}

def parse_mutations(
    response: str,
    parse_input: bool = True,
    parse_output: bool = True,
    remove_after_return: bool = False,
    remove_comments: bool = False,
    remove_print: bool = False,
    reject_multiple_functions: bool = True,
    reject_test_input_in_code: bool = False,
    f_replace_location: str = 'not_first',
    ) -> Tuple[bool, Dict[str, Any]]:
    """
    Parse the JSON response produced by the `code_output_mutation_prompt`.

    Returns
    -------
    (success, data)
        success : bool   – overall parsing success
        data    : dict   – keyed by variant ("variant_1", "variant_2", ...)
                           Each value is a dict with:
                               - complexity_attributes : list[str]
                               - description           : str
                               - code                  : str (cleaned)
                               - input                 : str
                               - imports               : list[str]
    """
    # # 1. Clean out <think> … </think> and ```json fences
    # cleaned = strip_json(strip_think_token(response))

    # 2. Load JSON
    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        return False, {}

    parsed: Dict[str, Any] = {}

    # 3. Iterate over variants
    for variant, content in obj.items():
        code_block  = content.get("code", "").strip()
        input_block = content.get("input", "").strip()

        # Combine code + input so `parse_code_input_output` can work as-is
        combined = f"{code_block}\n\n{input_block}"

        ok, code_dict = parse_code_input_output(
            combined,
            parse_input=parse_input,
            parse_output=parse_output,
            remove_after_return=remove_after_return,
            remove_comments=remove_comments,
            remove_print=remove_print,
            reject_multiple_functions=reject_multiple_functions,
            reject_test_input_in_code=reject_test_input_in_code,
            f_replace_location=f_replace_location,
        )
        if not ok:
            # Bail out on first failure
            return False, {}

        parsed[variant] = {
            "complexity_attributes": content.get("complexity_attributes", []),
            "description": content.get("description", ""),
            **code_dict,   # adds keys: code, input, imports
        }

    return True, parsed


def parse_curriculum(response: str,
    parse_input: bool = True,
    parse_output: bool = True,
    remove_after_return: bool = False,
    remove_comments: bool = False,
    remove_print: bool = False,
    reject_multiple_functions: bool = True,
    reject_test_input_in_code: bool = False,
    f_replace_location: str = 'not_first',
    ) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Parse the JSON response produced by the curriculum generation prompt.

    Expected schema:
    {
        "tasks": [
            {
                "code": "```<code snippet>```",
                "input": "```input\\n<task input>\\n```",
                "skills": ["skill1", "skill2", ...]
            },
            ...
        ]
    }
    """
    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        return False, []

    tasks = obj.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return False, []

    parsed_tasks: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            return False, []

        code_block = task.get("code", "")
        input_block = task.get("input", "")
        skills = task.get("skill", [])
        reasoning = task.get("reasoning", "")

        if not isinstance(skills, list) or not all(isinstance(s, str) and s.strip() for s in skills):
            return False, []

        combined = f"{code_block}\n\n{input_block}"

        ok, code_dict = parse_code_input_output(
            combined,   
            parse_input=parse_input,
            parse_output=parse_output,
            remove_after_return=remove_after_return,
            remove_comments=remove_comments,
            remove_print=remove_print,
            reject_multiple_functions=reject_multiple_functions,
            reject_test_input_in_code=reject_test_input_in_code,
            f_replace_location=f_replace_location,
        )

        if not ok:
            return False, []

        parsed_tasks.append(
            {
                "skill": skills,
                "detected_skill": skills,
                "reasoning": reasoning,
                **code_dict,
            }
        )

    return True, parsed_tasks

def parse_crossover(
    response: str,
    parse_input: bool = True,
    parse_output: bool = True,
    remove_after_return: bool = False,
    remove_comments: bool = False,
    remove_print: bool = False,
    reject_multiple_functions: bool = True,
    reject_test_input_in_code: bool = False,
    f_replace_location: str = 'not_first',
) -> Tuple[bool, Dict[str, Any]]:
    """Parse the JSON response produced by the `code_output_crossover_prompt`."""

    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        return False, {}

    code_block = obj.get("code", "").strip()
    input_block = obj.get("input", "").strip()

    if not code_block:
        return False, {}

    combined = f"{code_block}\n\n{input_block}" if input_block else code_block

    ok, code_dict = parse_code_input_output(
        combined,
        parse_input=parse_input,
        parse_output=parse_output,
        remove_after_return=remove_after_return,
        remove_comments=remove_comments,
        remove_print=remove_print,
        reject_multiple_functions=reject_multiple_functions,
        reject_test_input_in_code=reject_test_input_in_code,
        f_replace_location=f_replace_location,
    )

    if not ok:
        return False, {}

    parsed = {
        "skill_combination": obj.get("skill_combination", []),
        "crossover_description": obj.get("crossover_description", ""),
        **code_dict,
    }

    return True, parsed



if __name__ == "__main__":
    with open("res.json", "r") as f:
        result = f.read()

    success, variants = parse_mutations(
        result,
        parse_input=True,
        parse_output=False,
        remove_after_return=False,
        remove_comments=False,
        remove_print=False,
        reject_multiple_functions=False,
        reject_test_input_in_code=False,
        f_replace_location='not_first',
    )

    for variant, content in variants.items():
        for key, value in content.items():
            print(key, value)
        print("-"*100)