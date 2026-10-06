import ast
from dataclasses import dataclass, asdict


@dataclass
class CodeMetrics:
    loc: int
    functions: int
    loops: int
    conditionals: int
    complexity: int
    nesting: int

    def to_dict(self):
        return asdict(self)


class MetricVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = 0
        self.loops = 0
        self.conditionals = 0
        self.complexity = 1
        self.current_nesting = 0
        self.max_nesting = 0

    def _enter_block(self):
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)

    def _leave_block(self):
        self.current_nesting -= 1

    def visit_FunctionDef(self, node):
        self.functions += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_AsyncFunctionDef(self, node):
        self.functions += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_For(self, node):
        self.loops += 1
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_AsyncFor(self, node):
        self.loops += 1
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_While(self, node):
        self.loops += 1
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_If(self, node):
        self.conditionals += 1
        self.complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_IfExp(self, node):
        self.conditionals += 1
        self.complexity += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.complexity += len(node.handlers)
        self._enter_block()
        self.generic_visit(node)
        self._leave_block()

    def visit_BoolOp(self, node):
        # Each extra boolean branch increases decision complexity.
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)


def _count_loc(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith('#'))


def extract_metrics(code: str) -> CodeMetrics:
    if not code or not code.strip():
        raise ValueError("Source code is empty.")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc.msg} at line {exc.lineno}") from exc

    visitor = MetricVisitor()
    visitor.visit(tree)
    return CodeMetrics(
        loc=_count_loc(code),
        functions=visitor.functions,
        loops=visitor.loops,
        conditionals=visitor.conditionals,
        complexity=visitor.complexity,
        nesting=visitor.max_nesting,
    )
