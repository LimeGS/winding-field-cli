"""Explicit CLI/config resolution; scientific schemas remain with their validators."""
import argparse
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

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
    from .tifxyz_contract import parse_json
    result=parse_json(Path(path).read_text())
    if not isinstance(result,dict):raise ValueError('settings must be a JSON object')
    return result


def _config_arguments(parser):
    for action in list(parser._actions):
        if isinstance(action,(argparse._StoreTrueAction,argparse._StoreFalseAction)):
            for option in action.option_strings:
                if not option.startswith('--'):continue
                inverse='--'+option[5:] if option.startswith('--no-') else '--no-'+option[2:]
                if inverse not in parser._option_string_actions:
                    parser.add_argument(inverse,dest=action.dest,default=argparse.SUPPRESS,
                        action='store_false' if action.const else 'store_true',
                        help='explicit '+str(not action.const).lower()+' override for '+action.dest)
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


def _remainder(parser,tokens):
    if not any(a.nargs==argparse.REMAINDER for a in parser._actions):return tokens,[]
    index=0
    while index<len(tokens):
        token=tokens[index]
        action=parser._option_string_actions.get(token.split('=',1)[0])
        if token=='--' or action is None:return tokens[:index],tokens[index:]
        count=0 if '=' in token or action.nargs==0 else 1 if action.nargs in (None,'?') else action.nargs
        if not isinstance(count,int):raise ValueError('variable wrapper options before command remainder are unsupported')
        index+=1+count
    return tokens,[]


def _value(action,value):
    if isinstance(action,(argparse._StoreTrueAction,argparse._StoreFalseAction,argparse.BooleanOptionalAction)):
        if type(value) is not bool:raise ValueError(action.dest+' must be boolean')
        return value
    if value is None:
        if action.required:raise ValueError(action.dest+' is required')
        return None
    def scalar(value,converter=action.type):
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
            types=getattr(action,'element_types',None)
            return [scalar(v,types[i]) if types else scalar(v) for i,v in enumerate(value)]
        return scalar(value)
    if isinstance(action,argparse._AppendAction):
        if not isinstance(value,list):raise ValueError(action.dest+' must be a list')
        return [item(v) for v in value]
    return item(value)


def parse_options(parser,argv=None,*,paths=(),lexical_paths=(),sources=None):
    """Use typed argparse defaults, not JSON-to-argv translation or a new schema."""
    tokens=list(sys.argv[1:] if argv is None else argv)
    _config_arguments(parser)
    tokens,suffix=_remainder(parser,tokens)
    reader=argparse.ArgumentParser(add_help=False)
    reader.add_argument('--cli-config')
    _,remaining=reader.parse_known_args(tokens)
    parents=_selected(parser,remaining);active=parents[-1]
    explicit={t.split('=',1)[0] for t in remaining[:remaining.index('--') if '--' in remaining else len(remaining)]}
    constants={}
    for p in parents:
        for action in p._actions:
            if isinstance(action,argparse.BooleanOptionalAction) and len(explicit.intersection(action.option_strings))>1:
                parser.error('contradictory flags for '+action.dest)
            if isinstance(action,(argparse._StoreTrueAction,argparse._StoreFalseAction)) and explicit.intersection(action.option_strings):
                constants.setdefault(action.dest,set()).add(action.const)
    if any(len(values)>1 for values in constants.values()):parser.error('contradictory boolean flags')
    aliases=['--cli-config']
    if active._option_string_actions.get('--config') is active._option_string_actions.get('--cli-config'):
        aliases.append('--config')
    reader=argparse.ArgumentParser(add_help=False)
    reader.add_argument(*aliases,dest='file')
    selected,remaining=reader.parse_known_args(tokens)
    if selected.file is None:return parser.parse_args([*tokens,*suffix])
    if '--help' in tokens or '-h' in tokens:return parser.parse_args([*tokens,*suffix])
    actions={}
    for p in parents:
        for a in p._actions:
            if not isinstance(a,(argparse._HelpAction,argparse._VersionAction,argparse._SubParsersAction)) and a.dest!='_cli_config':
                actions.setdefault(a.dest,a)
    snapshots=[];groups=[]
    try:
        source=Path(selected.file).expanduser().resolve()
        raw=source.read_bytes()
        from .tifxyz_contract import parse_json
        settings=parse_json(raw.decode())
        if not isinstance(settings,dict):raise ValueError('settings must be a JSON object')
        if sources is not None:sources[str(source)]=hashlib.sha256(raw).hexdigest()
        if set(settings)-set(actions):raise ValueError('unknown settings: '+', '.join(sorted(set(settings)-set(actions))))
        for name,value in settings.items():
            action=actions[name];value=_value(action,value)
            positions=[p[1] for p in paths if isinstance(p,tuple) and p[0]==name]
            if value is not None and (action.type is Path or name in paths or name in lexical_paths or positions):
                def absolute(v):
                    if str(v).startswith(('https://','http://','s3://')):return v
                    result=source.parent/Path(v).expanduser()
                    result=result.absolute() if name in lexical_paths else result.resolve()
                    return result if action.type is Path else str(result)
                if positions:
                    for row in value if isinstance(action,argparse._AppendAction) else [value]:
                        for index in positions:row[index]=absolute(row[index])
                else:value=[absolute(v) for v in value] if isinstance(value,list) else absolute(value)
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
        args=parser.parse_args([*remaining,*suffix])
        if not suffix:
            for name,action in actions.items():
                if action.nargs==argparse.REMAINDER and name in settings and getattr(args,name)==[]:
                    setattr(args,name,deepcopy(action.default))
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


def request_snapshot(output: Path, request: dict, *, sources=None):
    """Bind immutable sibling recipes without occupying a solver-owned directory."""
    import json
    output=Path(output).resolve()
    envelope=dict(schema='winding-field-cli-request-v1',output=str(output),request=request,sources=sources or {})
    raw=(json.dumps(envelope,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()
    path=output.with_name(output.name+'.request-'+hashlib.sha256(raw).hexdigest()+'.json')
    return path,raw


def persist_request(output: Path, request: dict, *, resume: bool, sources=None) -> Path:
    path,raw=request_snapshot(output,request,sources=sources)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('saved request changed')
        return path
    if resume:raise ValueError('saved request is missing')
    # Publish a complete fsynced inode exclusively; never overwrite another run's recipe.
    with NamedTemporaryFile(dir=path.parent,prefix='.'+path.name,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
            try:os.link(temporary,path)
            except FileExistsError:
                if path.read_bytes()!=raw:raise ValueError('saved request changed')
        finally:
            temporary.unlink()
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return path


def field_arguments(parser,*,descriptor=True):
    """Shared field descriptor and WORLD geometry, deliberately separate from CT."""
    group=parser.add_argument_group('field source and WORLD geometry',argument_default=argparse.SUPPRESS)
    if descriptor:
        for name in ('kind','url','sha256','complete_sha256','support_policy'):
            group.add_argument('--field-'+name.replace('_','-'))
        for name in ('path','mask'):
            group.add_argument('--field-'+name,type=Path)
        group.add_argument('--field-mask-sha256')
    else:group.add_argument('--field-sha256')
    group.add_argument('--field-frame')
    group.add_argument('--field-voxel-um',type=float,help='field voxel calibration in micrometres, not CT sampling')
    for name,value_type in (('origin_xyz',float),('spacing_xyz',float),('shape_zyx',int)):
        group.add_argument('--field-'+name.replace('_','-'),type=value_type,nargs=3,
                           help='WORLD XYZ origin/spacing or integer ZYX node counts')
    group.add_argument('--sheet-step',type=float,help='raw field units per sheet')
    group.add_argument('--calibration')


def field_overrides(args):
    values=vars(args);result={}
    for name in ('kind','path','url','sha256','complete_sha256','support_policy'):
        if 'field_'+name in values:
            value=values['field_'+name]
            result.setdefault('field',{})[name]=str(value.resolve()) if isinstance(value,Path) else value
    mask={}
    if 'field_mask' in values:mask['path']=str(args.field_mask.resolve())
    if 'field_mask_sha256' in values:mask['sha256']=args.field_mask_sha256
    if mask:result.setdefault('field',{})['mask']=mask
    for name in ('frame','voxel_um','origin_xyz','spacing_xyz','shape_zyx','sheet_step','calibration'):
        flag=name if name in ('sheet_step','calibration') else 'field_'+name
        if flag in values:result.setdefault('geometry',{})[name]=values[flag]
    return result
