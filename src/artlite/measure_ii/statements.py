"""The loop statements of the source files the remarks name: their location, extent and trip count."""

from pathlib import Path

from ..cxx_scan import Statement, loop_statements, tokenize
from .report import SourceLoop
from .trips import parse_source_trip


def _loop_statement(statements: list[Statement], line: int, column: int) -> Statement | None:
    """The loop statement of a remark location.

    The location is the statement's keyword, unless the loop's preheader has
    no location: then it is the header's branch, inside the statement."""
    for statement in statements:
        if statement.line == line and (statement.column == column or not column):
            return statement
    around = [
        statement
        for statement in statements
        if (statement.line, statement.column) <= (line, column) and line <= statement.end_line
    ]
    return max(around, key=lambda statement: (statement.line, statement.column), default=None)


def _around(statements: list[Statement], loop: SourceLoop) -> Statement | None:
    """The innermost loop statement strictly around a located source loop."""
    if loop.end_line is None:
        return None
    around = [
        statement
        for statement in statements
        if (statement.line, statement.column) < (loop.line, loop.column) and statement.end_line >= loop.end_line
    ]
    return max(around, key=lambda statement: (statement.line, statement.column), default=None)


def locate(loop: SourceLoop, statement: Statement) -> None:
    loop.line, loop.column, loop.end_line, loop.exits, loop.scope = (
        statement.line,
        statement.column,
        statement.end_line,
        statement.exits,
        statement.scope,
    )


def read_statements(loops: list[SourceLoop], *, source: Path) -> None:
    """Locate each source loop's statement: keyword, last line and constant trip count.

    An outer loop whose body starts with an inner loop can report the inner loop
    statement's location (its preheader's branch has it): of the loops at one
    statement, one with more blocks encloses the others and takes the loop
    statement around them (copies of one loop have the same block count)."""
    statements_by_path: dict[str, list[Statement] | None] = {}
    for loop in loops:
        if loop.path not in statements_by_path:
            try:
                text = _source_path(loop.path, source=source).read_text(errors="replace")
                statements_by_path[loop.path] = loop_statements(tokenize(text))
            except OSError:
                statements_by_path[loop.path] = None
        statements = statements_by_path[loop.path]
        statement = _loop_statement(statements, loop.line, loop.column) if statements is not None else None
        if statement is not None:
            locate(loop, statement)
    at_statement: dict[tuple[str, int, int], list[SourceLoop]] = {}
    for loop in loops:
        if loop.end_line is not None:
            at_statement.setdefault((loop.path, loop.line, loop.column), []).append(loop)
    for (path, _, _), group in at_statement.items():
        sizes = sorted({loop.num_blocks or 1 for loop in group})
        for loop in group:
            for _ in range(sizes.index(loop.num_blocks or 1)):
                around = _around(statements_by_path[path] or [], loop)
                if around is None:
                    break
                locate(loop, around)
    for loop in loops:
        if statements_by_path[loop.path] is not None:
            loop.trip = parse_source_trip(_source_path(loop.path, source=source), loop.line)


def _source_path(path: str, *, source: Path) -> Path:
    """A DebugLoc file: the measured source when it names it relatively (as the compile command did)."""
    candidate = Path(path)
    return source if candidate.name == source.name and not candidate.is_absolute() else candidate
