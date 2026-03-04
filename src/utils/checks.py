import hashlib
import ast
import re
from typing import List

def check_determinism(code: str, inputs: str, executor, prev_output: str = None, n_runs: int = 1):
    """expects an executor that outputs string output and status"""
    all_outputs = set()
    if prev_output is not None:
        hash = hashlib.md5(str(prev_output).encode()).hexdigest()
        all_outputs.add(hash)
    for _ in range(n_runs):
        result = executor.run_code(code, inputs)[0]
        hash = hashlib.md5(str(result).encode()).hexdigest()
        all_outputs.add(hash)
    return len(all_outputs) == 1

def contains_banned_imports(code: str, banned_keywords: List[str], banned_keywords_for_errors_and_exceptions: List[str] = []) -> bool:
    """Check if code imports any banned modules using AST parsing."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(banned in alias.name.split('.') for banned in banned_keywords):
                        return True
            elif isinstance(node, ast.ImportFrom):
                module = node.module.split('.') if node.module else []
                if any(banned in module for banned in banned_keywords):
                    return True
                for alias in node.names:
                    if any(banned in alias.name.split('.') for banned in banned_keywords):
                        return True

            if banned_keywords_for_errors_and_exceptions:
                # Check for assert statements
                if isinstance(node, ast.Assert) and 'assert' in banned_keywords_for_errors_and_exceptions:
                    return True

                # Check for raise statements
                elif isinstance(node, ast.Raise) and 'raise' in banned_keywords_for_errors_and_exceptions:
                    return True

                # Check for try-except blocks
                elif isinstance(node, ast.Try) and 'try' in banned_keywords_for_errors_and_exceptions:
                    return True

                # Check for except handlers
                elif isinstance(node, ast.ExceptHandler) and 'except' in banned_keywords_for_errors_and_exceptions:
                    return True

        return False
    except SyntaxError:
        # Fallback to simple check if AST parsing fails
        return any(re.search(rf'\b{re.escape(banned)}\b', code) for banned in banned_keywords)
    

def check_no_definitions(code: str, composite_functions: List[str]) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in composite_functions:
            return False
    return True