"""Deterministic documentation and per-module AST audit."""
import argparse
import ast
import json
from pathlib import Path
import sys

from . import __version__
from .phrases import INPAINTING, NOINPAINTING, DIRECT_NAMES, GENERIC_NAMES, PIPELINE_CLASSES


def prose(text):
    result = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            fenced = not fenced
        elif not fenced:
            result.append(line.replace('`', ''))
    return '\n'.join(result).lower()


def match_phrases(text, phrases):
    """Return longest-first nonoverlapping matches and remaining text."""
    hits = []
    for phrase in sorted(phrases, key=len, reverse=True):
        while phrase in text:
            start = text.index(phrase)
            hits.append(phrase)
            text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
    return hits, text


def claims(text):
    negative, remaining = match_phrases(prose(text), NOINPAINTING)
    positive, _ = match_phrases(remaining, INPAINTING)
    return positive, negative


def evidence(source):
    tree = ast.parse(source)
    aliases = {}
    shadowed = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(node.name)
        elif isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or '') + '.' + item.name

    def resolve(node):
        if isinstance(node, ast.Name):
            if node.id in shadowed:
                return ''
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    def allowed(name, original):
        return name and name.split('.')[-1] not in shadowed and original not in shadowed

    def construction(node):
        if not isinstance(node, ast.Call):
            return False
        target = node.func
        if isinstance(target, ast.Attribute) and target.attr == 'from_pretrained':
            target = target.value
        name = resolve(target)
        return bool(allowed(name, '') and name.split('.')[-1] in PIPELINE_CLASSES)

    instances = set()
    # Fixed point supports aliases of constructed instances without cross-file inference.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                is_instance = construction(value) or (isinstance(value, ast.Name) and value.id in instances)
                if is_instance:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id not in instances:
                            instances.add(target.id)
                            changed = True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if construction(target):
            return True
        if isinstance(target, ast.Name) and target.id in instances and target.id not in shadowed:
            return True
        name = resolve(target)
        if not allowed(name, getattr(target, 'id', '')):
            continue
        leaf = name.split('.')[-1]
        if leaf in DIRECT_NAMES + GENERIC_NAMES:
            # Bare names require a from-import (or imported alias); attributes may use any base.
            if isinstance(target, ast.Attribute) or (isinstance(target, ast.Name) and target.id in aliases):
                return True
    return False


EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'virtualenv'}


def audit(repo, ignore=(), verbose=False):
    repo = Path(repo)
    if not repo.is_dir():
        raise OSError('Repository is not a directory: ' + str(repo))
    decisions = []
    positive = []
    negative = []
    test_positive = []
    supported = False
    for path in sorted(repo.rglob('*')):
        rel = path.relative_to(repo)
        name = rel.as_posix()
        if any(part in EXCLUDED for part in rel.parts) or any((repo / parent / 'pyvenv.cfg').is_file() for parent in rel.parents if str(parent) != '.'):
            continue
        if any(name.startswith(prefix) for prefix in ignore):
            decisions.append(name + ': ignored')
            continue
        if not path.is_file():
            continue
        documentation = name == 'README.md' or (rel.parts[0] == 'docs' and path.suffix == '.md')
        fixture = 'tests' in rel.parts[:-1]
        if path.suffix == '.py':
            source = path.read_text(encoding='utf-8')
            try:
                found = evidence(source)
            except SyntaxError:
                found = False
                decisions.append(name + ': invalid syntax, no AST evidence')
            supported = supported or found
            decisions.append(name + (': evidence' if found else ': no evidence'))
        if documentation or fixture:
            yes, no = claims(path.read_text(encoding='utf-8')) if path.suffix in {'.md', '.py', '.txt', '.rst'} else ([], [])
            if documentation:
                positive.extend(yes)
                negative.extend(no)
            elif fixture:
                test_positive.extend(yes)
            decisions.append(name + ': prose scanned')
    findings = []
    if positive and not supported:
        findings.append({'rule': 'documented-image-inpainting-claim-false', 'severity': 'HIGH', 'verdict': 'FAIL', 'message': 'Documentation claims the checked feature, but no source evidence was found.'})
    elif test_positive and not positive and not supported:
        findings.append({'rule': 'documented-image-inpainting-claim-false', 'severity': 'MEDIUM', 'verdict': 'FAIL', 'message': 'Only test fixtures claim the checked feature; no source evidence was found.'})
    if negative and supported:
        findings.append({'rule': 'documented-no-image-inpainting-claim-false', 'severity': 'HIGH', 'verdict': 'FAIL', 'message': 'Documentation denies the checked feature, but source evidence was found.'})
    return findings, decisions


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(prog='image-inpainting-claim-audit')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    run = sub.add_parser('audit', help='Audit local documentation and source')
    run.add_argument('repo', nargs='?', default='.')
    run.add_argument('--json', action='store_true')
    run.add_argument('--verbose', action='store_true')
    run.add_argument('--ignore', action='append', default=[], metavar='PREFIX')
    run.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    args = parser.parse_args(argv)
    try:
        findings, decisions = audit(args.repo, args.ignore, args.verbose)
    except (OSError, UnicodeError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 1
    if args.verbose:
        print('Scope: README.md, docs/**/*.md, tests fixtures; all eligible *.py modules in ' + str(Path(args.repo).resolve()), file=sys.stderr)
        for decision in decisions:
            print(decision, file=sys.stderr)
    failed = any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings)
    if args.json:
        print(json.dumps({'verdict': 'FAIL' if failed else 'PASS', 'findings': findings}))
    elif findings:
        for finding in findings:
            print('{severity} {rule}: {message}'.format(**finding))
    else:
        print('PASS: no contradictory claims found.')
    return 2 if failed else 0
