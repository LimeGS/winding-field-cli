import argparse
import json
from pathlib import Path

import pytest


def test_false_zero_and_omission_are_distinct():
    from winding_field.cli_settings import merge_explicit
    base={'render':{'previews':True,'cache_bytes':64,'width':10},'pitch':2}
    assert merge_explicit(base,{'render':{'previews':False,'cache_bytes':0},'pitch':None})=={
        'render':{'previews':False,'cache_bytes':0,'width':10},'pitch':None}
    assert base['render']['previews'] is True and base['pitch']==2


def test_nested_lists_replace_without_reordering():
    from winding_field.cli_settings import merge_explicit,request_sha256
    base={'levels':[-1,-2],'nested':{'points':[[1,2,3]]}}
    overrides={'levels':[-3],'nested':{'points':[]}}
    result=merge_explicit(base,overrides)
    assert result=={'levels':[-3],'nested':{'points':[]}}
    result['levels'].append(-4)
    assert overrides['levels']==[-3] and base['levels']==[-1,-2]
    assert request_sha256({'b':2,'a':1})==request_sha256({'a':1,'b':2})


@pytest.mark.parametrize('text',['{"x": NaN}','{"x": 1, "x": 2}','[]'])
def test_nonfinite_and_duplicate_metadata_reject(tmp_path,text):
    from winding_field.cli_settings import read_settings
    path=tmp_path/'invalid.json';path.write_text(text)
    with pytest.raises(ValueError):read_settings(path)


def test_file_and_flag_paths_use_different_bases(tmp_path,monkeypatch):
    from winding_field.cli_settings import parse_options
    file_dir=tmp_path/'file base';file_dir.mkdir()
    cwd=tmp_path/'caller base';cwd.mkdir();monkeypatch.chdir(cwd)
    config=file_dir/'settings.json'
    config.write_text(json.dumps({'input':'same.bin','output':'result','width':17}))
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--width',type=int,default=64)
    args=parse_options(parser,['--config',str(config)],paths=('output',))
    assert args.input==file_dir/'same.bin' and args.output==str(file_dir/'result')
    assert args.width==17
    args=parse_options(parser,['--config',str(config),'--input','override.bin','--width','0'],paths=('output',))
    assert args.input.resolve()==cwd/'override.bin' and args.width==0


def test_config_supplies_required_positionals_and_boolean_false(tmp_path):
    from winding_field.cli_settings import parse_options
    config=tmp_path/'settings.json'
    config.write_text(json.dumps({'run':'existing','enabled':False,'levels':[-3,-1]}))
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--enabled',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--levels',nargs='+',type=float,required=True)
    args=parse_options(parser,['--config',str(config)])
    assert args.run==tmp_path/'existing'
    assert args.enabled is False and args.levels==[-3.,-1.]
    args=parse_options(parser,['--config',str(config),'different','--enabled'])
    assert args.run==Path('different') and args.enabled is True


def test_config_does_not_persist_defaults_on_reused_parser(tmp_path):
    from winding_field.cli_settings import parse_options
    config=tmp_path/'settings.json';config.write_text('{"n": 17}')
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=64)
    assert parse_options(parser,['--config',str(config)]).n==17
    assert parse_options(parser,[]).n==64


def test_subcommand_file_defaults_do_not_change_command(tmp_path):
    from winding_field.cli_settings import parse_options
    config=tmp_path/'settings.json';config.write_text('{"source": "data", "count": 3}')
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect');p.add_argument('source');p.add_argument('--count',type=int,default=1)
    args=parse_options(parser,['inspect','--config',str(config)])
    assert args.command=='inspect' and args.source=='data' and args.count==3
    config.write_text('{"command": "other", "source": "data"}')
    with pytest.raises(SystemExit):parse_options(parser,['inspect','--config',str(config)])


@pytest.mark.parametrize('mapping',[{'missing':1},{'n':True},{'n':0.5},{'flag':'false'},
                                    {'mode':'unknown'},{'vector':[1,2]},{'vector':[1,2,float('inf')]}])
def test_config_cannot_bypass_argument_types_and_choices(tmp_path,mapping):
    from winding_field.cli_settings import parse_options
    config=tmp_path/'settings.json';config.write_text(json.dumps(mapping))
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=1);parser.add_argument('--flag',action='store_true')
    parser.add_argument('--mode',choices=['known'],default='known')
    parser.add_argument('--vector',nargs=3,type=float)
    with pytest.raises(SystemExit):parse_options(parser,['--config',str(config)])


def test_file_and_flag_conflicting_sources_are_rejected(tmp_path):
    from winding_field.cli_settings import parse_options
    config=tmp_path/'settings.json';config.write_text('{"field": "a"}')
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--field');group.add_argument('--mesh')
    assert parse_options(parser,['--config',str(config)]).field=='a'
    with pytest.raises(SystemExit):parse_options(parser,['--config',str(config),'--mesh','b'])


def test_legacy_config_flag_is_not_intercepted(tmp_path):
    from winding_field.cli_settings import parse_options
    legacy=tmp_path/'native.json';legacy.write_text('{"schema": "native"}')
    cli_config=tmp_path/'options.json';cli_config.write_text('{"threads": 2}')
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=1)
    args=parse_options(parser,['--config',str(legacy),'--cli-config',str(cli_config)])
    assert args.config==legacy and args.threads==2


def test_repeated_flags_replace_the_file_list_in_input_order(tmp_path):
    from winding_field.cli_settings import parse_options
    config=tmp_path/'settings.json';config.write_text('{"point": [[1, 2, 3]]}')
    parser=argparse.ArgumentParser()
    parser.add_argument('--point',action='append',nargs=3,type=float)
    args=parse_options(parser,['--config',str(config),'--point','4','5','6','--point','7','8','9'])
    assert args.point==[[4.,5.,6.],[7.,8.,9.]]
    config.write_text('{"point": []}')
    assert parse_options(parser,['--config',str(config)]).point==[]


def test_contradictory_positive_and_negative_flags_reject():
    from winding_field.cli_settings import parse_options
    parser=argparse.ArgumentParser()
    parser.add_argument('--enabled',action=argparse.BooleanOptionalAction,default=True)
    with pytest.raises(SystemExit):parse_options(parser,['--enabled','--no-enabled'])


@pytest.mark.parametrize('negative',[False,True])
def test_legacy_booleans_can_explicitly_override_both_file_values(tmp_path,negative):
    from winding_field.cli_settings import parse_options
    config=tmp_path/'cli.json'
    config.write_text(json.dumps(dict(enabled=not negative)))
    parser=argparse.ArgumentParser()
    if negative:parser.add_argument('--no-enabled',dest='enabled',action='store_false')
    else:parser.add_argument('--enabled',action='store_true')
    args=parse_options(parser,['--config',str(config),'--enabled' if negative else '--no-enabled'])
    assert args.enabled is negative
    with pytest.raises(SystemExit):parse_options(parser,['--enabled','--no-enabled'])
    assert parse_options(parser,[]).enabled is negative
