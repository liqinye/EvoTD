from typing import List

RUN_CODE_TEMPLATE = """{code}
repr(f({inputs}))"""

VALIDATE_CODE_TEMPLATE = """{code}
repr(f({inputs}))"""

EVAL_INPUT_PREDICTION_TEMPLATE = """{code}
from utils.equality import deep_equal as _deep_equal
_result = _deep_equal({gold_output}, f({agent_input}))
repr(_result)"""

EVAL_OUTPUT_PREDICTION_TEMPLATE = """{code}
from utils.equality import deep_equal as _deep_equal
_gold = eval({gold_output})
_agent = eval({agent_output})
_result = _deep_equal(_gold, _agent)
repr(_result)"""

CHECK_DETERMINISM_TEMPLATE = """{code}
returns = f({inputs})
if returns != f({inputs}):
    raise Exception('Non-deterministic code')
repr(returns)"""

def EVAL_K_INPUT_PREDICTION_TEMPLATE(code: str, gold_output: str, k_agent_inputs: List[str]):
    output_lines = [
        f"{code}",
        "from utils.equality import deep_equal as _deep_equal",
        "acc_list = []",
    ]
    for inp in k_agent_inputs:
        output_lines.append(
            f"try:\n    acc_list.append(_deep_equal({gold_output}, f({inp})))\nexcept Exception:\n    acc_list.append(False)"
        )
    output_lines.append("repr(acc_list)")
    return "\n".join(output_lines)

def EVAL_K_OUTPUT_PREDICTION_TEMPLATE(code: str, gold_output: str, k_agent_outputs: List[str]):
    output_lines = [
        f"{code}",
        "from utils.equality import deep_equal as _deep_equal",
        "acc_list = []",
    ]
    for out in k_agent_outputs:
        output_lines.append(
            f"try:\n    acc_list.append(_deep_equal({gold_output}, {out}))\nexcept Exception:\n    acc_list.append(False)"
        )
    output_lines.append("repr(acc_list)")
    return "\n".join(output_lines)
