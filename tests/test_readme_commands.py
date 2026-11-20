"""Every documented command uses flags that the current CLI accepts.

Examples in README.md and docs/*.md are parsed with the live parsers: each
`winding-field ...` or `wf ...` invocation is resolved to its (sub)command and
every `--flag` must appear in that command's help. Removing or renaming a flag
therefore fails here until the documentation follows.
"""

import contextlib
import functools
import io
import re
import shlex
from pathlib import Path

import pytest

from winding_field.cli import main

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
CHOICES = re.compile(r"\{([a-z0-9_,-]+)\}")
FLAG = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")


@functools.cache
def command_help(path):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        try:
            main([*path, "--help"])
        except SystemExit as exit:
            if exit.code not in (0, None):
                return None
    return output.getvalue()


def subcommands(path):
    text = command_help(path) or ""
    usage = text.split("\n\n", 1)[0]
    match = CHOICES.search(usage)
    return set(match.group(1).split(",")) if match else set()


def documented_commands():
    for document in DOCUMENTS:
        text = document.read_text()
        blocks = re.findall(r"```(?:bash|sh|console)?\n(.*?)```", text, re.DOTALL)
        lines = [line for block in blocks for line in re.sub(r"\\\n\s*", " ", block).splitlines()]
        lines += re.findall(r"`((?:winding-field|wf) [^`]+)`", text)
        for line in lines:
            match = re.match(r"\s*(?:winding-field|wf)\s+(.*)$", line)
            if match:
                yield document.relative_to(ROOT).as_posix(), match.group(1).split(" #")[0]


COMMANDS = sorted(set(documented_commands()))


def test_documentation_has_commands_to_check():
    readme = [command for document, command in COMMANDS if document == "README.md"]
    assert len(readme) >= 15


@pytest.mark.parametrize(("document", "command"), COMMANDS)
def test_documented_command_uses_current_flags(document, command):
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    path = ()
    for token in tokens:
        if token not in subcommands(path):
            break
        path += (token,)
    text = command_help(path)
    assert text is not None, f"{document}: `{' '.join(path)} --help` fails"
    known = set(FLAG.findall(text))
    unknown = [
        token.split("=", 1)[0]
        for token in tokens[len(path) :]
        if token.startswith("--") and token.split("=", 1)[0] not in known
    ]
    assert not unknown, f"{document}: `{' '.join(path)}` does not accept {unknown}: {command}"
