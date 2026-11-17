"""Explicit CLI/config resolution; scientific schemas remain with their validators."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys

from .tifxyz_contract import parse_json
from .util import digest


def merge_explicit(base: dict, overrides: dict) -> dict:
    result=deepcopy(base)
    for key,value in overrides.items():
        result[key]=(merge_explicit(result[key],value)
                     if isinstance(value,dict) and isinstance(result.get(key),dict) else deepcopy(value))
    return result


def request_sha256(request: dict) -> str:
    return digest(request)


def read_settings(path) -> dict:
    result=parse_json(Path(path).read_text())
    if not isinstance(result,dict):raise ValueError('settings must be a JSON object')
    return result


def _config_arguments(parser):
    if '--cli-config' not in parser._option_string_actions:
        names=['--cli-config']
        if '--config' not in parser._option_string_actions:names.append('--config')
        parser.add_argument(*names,dest='_cli_config',default=argparse.SUPPRESS,
            help='JSON defaults using argument names; explicit flags override; file paths are relative to this file')
    for action in parser._actions:
        if isinstance(action,argparse._SubParsersAction):
            for child in set(action.choices.values()):_config_arguments(child)


def _selected(parser,tokens):
    parents=[parser]
    while True:
        sub=next((a for a in parser._actions if isinstance(a,argparse._SubParsersAction)),None)
        if sub is None:break
        if not tokens or tokens[0] not in sub.choices:break
        parser=sub.choices[tokens[0]];parents.append(parser);tokens=tokens[1:]
    return parents


def _value(action,value):
    if isinstance(action,(argparse._StoreTrueAction,argparse._StoreFalseAction,argparse.BooleanOptionalAction)):
        if type(value) is not bool:raise ValueError(action.dest+' must be boolean')
        return value
    if value is None:
        if action.required:raise ValueError(action.dest+' is required')
        return None
    def scalar(value):
        converter=action.type
        if converter in (int,float):
            if isinstance(value,bool) or (converter is int and isinstance(value,float) and not value.is_integer()):
                raise ValueError('invalid number for '+action.dest)
        elif converter in (None,str,Path) and not isinstance(value,str):
            raise ValueError(action.dest+' must be a string')
        result=value if converter is None else converter(value)
        if isinstance(result,float) and not math.isfinite(result):raise ValueError('nonfinite '+action.dest)
        if action.choices is not None and result not in action.choices:raise ValueError('invalid choice for '+action.dest)
        return result
    def item(value):
        if action.nargs not in (None,'?'):
            if not isinstance(value,list):raise ValueError(action.dest+' must be a list')
            if isinstance(action.nargs,int) and len(value)!=action.nargs:raise ValueError('wrong arity for '+action.dest)
            if action.nargs=='+' and not value:raise ValueError(action.dest+' cannot be empty')
            return [scalar(v) for v in value]
        return scalar(value)
    if isinstance(action,argparse._AppendAction):
        if not isinstance(value,list):raise ValueError(action.dest+' must be a list')
        return [item(v) for v in value]
    return item(value)


def parse_options(parser,argv=None,*,paths=()):
    """Use typed argparse defaults, not JSON-to-argv translation or a new schema."""
    tokens=list(sys.argv[1:] if argv is None else argv)
    _config_arguments(parser)
    reader=argparse.ArgumentParser(add_help=False)
    reader.add_argument('--cli-config')
    _,remaining=reader.parse_known_args(tokens)
    parents=_selected(parser,remaining);active=parents[-1]
    explicit={t.split('=',1)[0] for t in remaining[:remaining.index('--') if '--' in remaining else len(remaining)]}
    for p in parents:
        for action in p._actions:
            if isinstance(action,argparse.BooleanOptionalAction) and len(explicit.intersection(action.option_strings))>1:
                parser.error('contradictory flags for '+action.dest)
    aliases=['--cli-config']
    if active._option_string_actions.get('--config') is active._option_string_actions.get('--cli-config'):
        aliases.append('--config')
    reader=argparse.ArgumentParser(add_help=False)
    reader.add_argument(*aliases,dest='file')
    selected,remaining=reader.parse_known_args(tokens)
    if selected.file is None:return parser.parse_args(tokens)
    if '--help' in tokens or '-h' in tokens:return parser.parse_args(tokens)
    actions={a.dest:a for p in parents for a in p._actions
             if not isinstance(a,(argparse._HelpAction,argparse._VersionAction,argparse._SubParsersAction))
             and a.dest!='_cli_config'}
    snapshots=[];groups=[]
    try:
        source=Path(selected.file).expanduser().resolve()
        settings=read_settings(source)
        if set(settings)-set(actions):raise ValueError('unknown settings: '+', '.join(sorted(set(settings)-set(actions))))
        for name,value in settings.items():
            action=actions[name];value=_value(action,value)
            if value is not None and (action.type is Path or name in paths):
                def absolute(v):
                    if str(v).startswith(('https://','http://','s3://')):return v
                    result=(source.parent/Path(v).expanduser()).resolve()
                    return result if action.type is Path else str(result)
                value=[absolute(v) for v in value] if isinstance(value,list) else absolute(value)
            snapshots.append((action,action.default,action.required,action.nargs))
            action.default=(None if isinstance(action,argparse._AppendAction)
                            and explicit.intersection(action.option_strings) else value)
            action.required=False
            if not action.option_strings:
                if action.nargs is None:action.nargs='?'
                elif action.nargs=='+':action.nargs='*'
        for p in parents:
            for group in p._mutually_exclusive_groups:
                groups.append((group,group.required))
                if any(a.dest in settings for a in group._group_actions):group.required=False
        args=parser.parse_args(remaining)
        for group,required in groups:
            present={a.dest for a in group._group_actions if getattr(args,a.dest,None) is not None}
            if len(present)>1 or (required and not present):
                raise ValueError('supply exactly one of '+', '.join(a.dest for a in group._group_actions))
        return args
    except (ValueError,TypeError,OSError) as error:
        parser.error(str(error))
    finally:
        for action,default,required,nargs in snapshots:
            action.default=default;action.required=required;action.nargs=nargs
        for group,required in groups:group.required=required
