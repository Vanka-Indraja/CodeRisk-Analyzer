import ast
from dataclasses import dataclass, asdict


@dataclass
class CodeMetrics:
    # ML features
    loc: int
    functions: int
    loops: int
    conditionals: int
    complexity: int
    nesting: int

    # Extra quality indicators
    classes: int
    imports: int
    exception_handlers: int
    comment_lines: int
    comment_ratio: float
    max_function_length: int
    avg_function_length: float
    long_functions: int

    def to_dict(self):
        return asdict(self)


class CodeAnalyzer(ast.NodeVisitor):

    def __init__(self):
        self.functions = 0
        self.loops = 0
        self.conditionals = 0
        self.classes = 0
        self.imports = 0
        self.exception_handlers = 0

        # Cyclomatic complexity starts at 1
        self.complexity = 1

        self.current_nesting = 0
        self.max_nesting = 0

        self.function_lengths = []

    def increase_nesting(self):
        self.current_nesting += 1

        self.max_nesting = max(
            self.max_nesting,
            self.current_nesting
        )

    def decrease_nesting(self):
        self.current_nesting -= 1

    # -------------------------
    # Functions
    # -------------------------

    def visit_FunctionDef(self, node):
        self.functions += 1

        if hasattr(node, "end_lineno") and node.end_lineno:
            length = node.end_lineno - node.lineno + 1
            self.function_lengths.append(length)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.functions += 1

        if hasattr(node, "end_lineno") and node.end_lineno:
            length = node.end_lineno - node.lineno + 1
            self.function_lengths.append(length)

        self.generic_visit(node)

    # -------------------------
    # Classes
    # -------------------------

    def visit_ClassDef(self, node):
        self.classes += 1
        self.generic_visit(node)

    # -------------------------
    # Imports
    # -------------------------

    def visit_Import(self, node):
        self.imports += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.imports += 1
        self.generic_visit(node)

    # -------------------------
    # Conditions
    # -------------------------

    def visit_If(self, node):
        self.conditionals += 1
        self.complexity += 1

        self.increase_nesting()
        self.generic_visit(node)
        self.decrease_nesting()

    def visit_IfExp(self, node):
        self.conditionals += 1
        self.complexity += 1
        self.generic_visit(node)

    # -------------------------
    # Loops
    # -------------------------

    def visit_For(self, node):
        self.loops += 1
        self.complexity += 1

        self.increase_nesting()
        self.generic_visit(node)
        self.decrease_nesting()

    def visit_AsyncFor(self, node):
        self.loops += 1
        self.complexity += 1

        self.increase_nesting()
        self.generic_visit(node)
        self.decrease_nesting()

    def visit_While(self, node):
        self.loops += 1
        self.complexity += 1

        self.increase_nesting()
        self.generic_visit(node)
        self.decrease_nesting()

    # -------------------------
    # Boolean logic
    # -------------------------

    def visit_BoolOp(self, node):
        extra_conditions = max(
            len(node.values) - 1,
            0
        )

        self.complexity += extra_conditions
        self.generic_visit(node)

    # -------------------------
    # Exception handling
    # -------------------------

    def visit_ExceptHandler(self, node):
        self.exception_handlers += 1
        self.complexity += 1

        self.increase_nesting()
        self.generic_visit(node)
        self.decrease_nesting()


def count_loc(source_code):
    count = 0

    for line in source_code.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        count += 1

    return count


def count_comments(source_code):
    comment_lines = 0

    for line in source_code.splitlines():
        if line.strip().startswith("#"):
            comment_lines += 1

    return comment_lines


def extract_metrics(source_code):

    if not source_code or not source_code.strip():
        raise ValueError(
            "Source code is empty."
        )

    try:
        tree = ast.parse(source_code)

    except SyntaxError as error:
        raise ValueError(
            f"Invalid Python code: {error.msg} "
            f"at line {error.lineno}"
        )

    analyzer = CodeAnalyzer()
    analyzer.visit(tree)

    loc = count_loc(source_code)

    comment_lines = count_comments(
        source_code
    )

    total_lines = len(
        source_code.splitlines()
    )

    if total_lines > 0:
        comment_ratio = round(
            (comment_lines / total_lines) * 100,
            2
        )
    else:
        comment_ratio = 0.0

    if analyzer.function_lengths:

        max_function_length = max(
            analyzer.function_lengths
        )

        avg_function_length = round(
            sum(analyzer.function_lengths)
            / len(analyzer.function_lengths),
            2
        )

        long_functions = sum(
            1
            for length in analyzer.function_lengths
            if length >= 30
        )

    else:

        max_function_length = 0
        avg_function_length = 0.0
        long_functions = 0

    return CodeMetrics(
        loc=loc,
        functions=analyzer.functions,
        loops=analyzer.loops,
        conditionals=analyzer.conditionals,
        complexity=analyzer.complexity,
        nesting=analyzer.max_nesting,

        classes=analyzer.classes,
        imports=analyzer.imports,
        exception_handlers=analyzer.exception_handlers,
        comment_lines=comment_lines,
        comment_ratio=comment_ratio,
        max_function_length=max_function_length,
        avg_function_length=avg_function_length,
        long_functions=long_functions
    )