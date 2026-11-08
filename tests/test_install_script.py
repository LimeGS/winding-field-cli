import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/install.sh'


@pytest.fixture
def installer(tmp_path):
    """Real Git/filesystem, without apt, network downloads or package builds."""
    home=tmp_path/'home';home.mkdir()
    work=tmp_path/'work with spaces';work.mkdir()
    remote=tmp_path/'remote';remote.mkdir()
    subprocess.run(['git','init','-q','--initial-branch=main',str(remote)],check=True)
    for name in ('pyproject.toml','uv.toml','docker/cli.in'):
        p=remote/name;p.parent.mkdir(exist_ok=True);p.write_text('# fixture\n')
    subprocess.run(['git','-C',str(remote),'add','.'],check=True)
    subprocess.run(['git','-C',str(remote),'-c','user.name=Test','-c','user.email=test@example.org',
                    'commit','-qm','fixture'],check=True)
    config=tmp_path/'gitconfig'
    subprocess.run(['git','config','--file',str(config),f'url.{remote.as_uri()}.insteadOf',
                    'https://github.com/LimeGS/winding-field-cli.git'],check=True)
    log=tmp_path/'calls.jsonl'
    bin=tmp_path/'bin';bin.mkdir()
    driver=bin/'driver'
    driver.write_text(f'#!{sys.executable}\n'+'''
import json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name;args=sys.argv[1:]
with open(os.environ['INSTALL_LOG'],'a') as f:f.write(json.dumps([name,*args])+'\\n')
if os.environ.get('INSTALL_FAIL')==name:sys.exit(42)
if name=='uv' and args[:2]==['pip','install']:
    stage='cli' if args[-1]=='.' else 'lasagna'
    if os.environ.get('INSTALL_FAIL_STAGE')==stage:sys.exit(43)
if name=='uname':print(os.environ.get('INSTALL_OS','Linux'))
elif name=='sudo':
    if args!=['-v']:os.execvp(args[0],args)
elif name=='python3.12':
    assert args[:2]==['-m','venv'];dest=Path(args[2]);(dest/'bin').mkdir(parents=True)
    (dest/'bin/python').symlink_to(Path(__file__).resolve())
elif name=='python':
    assert args[:3]==['-m','pip','install']
    tool=Path(sys.argv[0]).with_name('uv');tool.unlink(missing_ok=True)
    tool.symlink_to(Path(__file__).resolve())
elif name=='uv':
    if args[0]=='venv':
        dest=Path(args[-1]);(dest/'bin').mkdir(parents=True)
        (dest/'bin/python').symlink_to(Path(__file__).resolve())
    elif args[:2]==['pip','install'] and args[-1]=='.':
        entry=Path(sys.argv[0]).with_name('winding-field');entry.unlink(missing_ok=True)
        entry.symlink_to(Path(__file__).resolve())
elif name=='winding-field':assert args==['doctor']
elif name!='apt-get':raise AssertionError((name,args))
''')
    driver.chmod(0o755)
    for name in ('uname','sudo','apt-get','python3.12'):(bin/name).symlink_to(driver)
    env=dict(os.environ,HOME=str(home),PATH=str(bin)+os.pathsep+os.environ['PATH'],
             GIT_CONFIG_GLOBAL=str(config),GIT_CONFIG_NOSYSTEM='1',INSTALL_LOG=str(log))
    def run(*,content=None,**overrides):
        return subprocess.run(['bash',str(SCRIPT)] if content is None else ['bash'],
                              input=content,cwd=work,env=dict(env,**overrides),
                              capture_output=True,text=True,timeout=30)
    return run,work,home,log


def test_install_script_reuses_environments_and_preserves_dirty_checkout(installer):
    run,work,home,log=installer
    first=run();assert first.returncode==0,first.stderr
    checkout=work/'winding-field-cli'
    env=checkout/'.venv';lasagna=home/'.cache/winding-field/lasagna-venv'
    assert (env/'bin/winding-field').is_file() and (lasagna/'bin/python').is_file()
    before=(env/'bin/python').lstat().st_ino
    (checkout/'uv.toml').write_text('# user edit\n')
    (checkout/'runs').mkdir();(checkout/'runs/keep').write_text('checkpoint')
    second=run();assert second.returncode==0,second.stderr
    assert (env/'bin/python').lstat().st_ino==before
    assert (checkout/'uv.toml').read_text()=='# user edit\n'
    assert (checkout/'runs/keep').read_text()=='checkpoint'
    assert 'source ' in second.stdout and '.venv/bin/activate' in second.stdout
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    assert sum(call[:3]==['python3.12','-m','venv'] for call in calls)==1
    assert sum(call[:2]==['uv','venv'] for call in calls)==1
    assert any(call[:2]==['sudo','-v'] for call in calls)==(os.geteuid()!=0)


def test_install_script_recovers_missing_uv_without_recreating_venv(installer):
    run,work,_,_=installer
    first=run();assert first.returncode==0,first.stderr
    env=work/'winding-field-cli/.venv'
    before=(env/'bin/python').lstat().st_ino
    (env/'bin/uv').unlink()
    (env/'bin/winding-field').unlink()
    again=run();assert again.returncode==0,again.stderr
    assert (env/'bin/python').lstat().st_ino==before
    assert (env/'bin/winding-field').is_file()


def test_install_script_repairs_existing_outdated_uv(installer):
    run,work,_,_=installer
    first=run();assert first.returncode==0,first.stderr
    tool=work/'winding-field-cli/.venv/bin/uv'
    tool.unlink();tool.write_text('#!/bin/sh\necho "uv 0.7.5: unknown extra-build-variables" >&2\nexit 2\n')
    tool.chmod(0o755)
    result=run();assert result.returncode==0,result.stderr


@pytest.mark.parametrize('stage',['cli','lasagna'])
def test_install_script_recovers_after_package_install_failure(installer,stage):
    run,work,home,_=installer
    first=run(INSTALL_FAIL_STAGE=stage)
    assert first.returncode==43
    python=work/'winding-field-cli/.venv/bin/python'
    before=python.lstat().st_ino
    marker=work/'winding-field-cli/keep';marker.write_text('user data')
    lasagna=home/'.cache/winding-field/lasagna-venv/bin/python'
    lasagna_before=lasagna.lstat().st_ino if lasagna.exists() else None
    result=run();assert result.returncode==0,result.stderr
    assert python.lstat().st_ino==before and marker.read_text()=='user data'
    if lasagna_before is not None:assert lasagna.lstat().st_ino==lasagna_before


@pytest.mark.skipif(os.geteuid()==0,reason='root does not need sudo authorization')
def test_install_script_stops_when_sudo_authorization_fails(installer):
    run,work,_,log=installer
    result=run(INSTALL_FAIL='sudo')
    assert result.returncode==42
    assert not (work/'winding-field-cli').exists()
    assert not any(json.loads(line)[0]=='apt-get' for line in log.read_text().splitlines())


@pytest.mark.parametrize('kind',['directory','foreign-repository','broken-venv'])
def test_install_script_refuses_unrelated_or_broken_targets(installer,kind):
    run,work,_,_=installer
    target=work/'winding-field-cli';target.mkdir()
    marker=target/'keep';marker.write_text('user data')
    if kind!='directory':
        subprocess.run(['git','init','-q',str(target)],check=True)
        url='https://github.com/someone/other.git' if kind=='foreign-repository' else 'https://github.com/LimeGS/winding-field-cli.git'
        subprocess.run(['git','-C',str(target),'remote','add','origin',url],check=True)
    if kind=='broken-venv':(target/'.venv').mkdir()
    result=run()
    assert result.returncode!=0 and 'Refusing' in result.stderr
    assert marker.read_text()=='user data'


def test_install_script_rejects_unsupported_platform_before_installing(installer):
    run,work,_,log=installer
    result=run(INSTALL_OS='Darwin')
    assert result.returncode!=0 and 'Ubuntu' in result.stderr
    assert not (work/'winding-field-cli').exists()
    assert not any(json.loads(line)[0]=='apt-get' for line in log.read_text().splitlines())


def test_install_script_stops_on_dependency_failure(installer):
    run,work,_,_=installer
    result=run(INSTALL_FAIL='apt-get')
    assert result.returncode==42
    assert not (work/'winding-field-cli').exists()


def test_truncated_download_cannot_start_setup(installer):
    run,work,_,log=installer
    lines=SCRIPT.read_text().splitlines()
    result=run(content='\n'.join(lines[:len(lines)//2]))
    assert result.returncode!=0
    assert not log.exists() and not (work/'winding-field-cli').exists()
