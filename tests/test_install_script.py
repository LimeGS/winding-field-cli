import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/install.sh'
RUNTIME = ROOT / 'tools/install-runtime.sh'


@pytest.fixture
def installer(tmp_path):
    """Real Git/filesystem, without apt, network downloads or package builds."""
    home=tmp_path/'home';home.mkdir()
    work=tmp_path/'work with spaces';work.mkdir()
    remote=tmp_path/'remote';remote.mkdir()
    subprocess.run(['git','init','-q','--initial-branch=main',str(remote)],check=True)
    for name in ('pyproject.toml','uv.toml','docker/cli.in'):
        p=remote/name;p.parent.mkdir(exist_ok=True);p.write_text('# fixture\n')
    (work/'pyproject.toml').write_text('[project]\nname="fixture"\nversion="0.0.1"\n')
    for name in ('tools/install-runtime.sh','tools/ubuntu-packages.txt','docker/lasagna.in','docker/build-constraints.txt'):
        if (ROOT/name).exists():
            p=remote/name;p.parent.mkdir(exist_ok=True);shutil.copyfile(ROOT/name,p)
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
import json,os,shlex,sys
from pathlib import Path
name=Path(sys.argv[0]).name;args=sys.argv[1:]
with open(os.environ['INSTALL_LOG'],'a') as f:f.write(json.dumps([name,*args])+'\\n')
if os.environ.get('INSTALL_FAIL')==name:sys.exit(42)
if name=='uv' and args[0]=='build' and os.environ.get('INSTALL_FAIL_STAGE')=='build':sys.exit(43)
if name=='uv' and args[:2]==['pip','check'] and os.environ.get('INSTALL_FAIL_STAGE')=='check':sys.exit(43)
if name=='uv' and args[:2] in (['pip','install'],['pip','sync']):
    target=Path(args[args.index('--python')+1]) if '--python' in args else Path(sys.argv[0]).with_name('python')
    stage='lasagna' if 'lasagna' in str(target) else 'cli'
    if os.environ.get('INSTALL_FAIL_STAGE')==stage:sys.exit(43)
if name=='uname':print(os.environ.get('INSTALL_OS','Linux'))
elif name=='curl':
    sys.stdout.write(Path(os.environ['INSTALL_SCRIPT']).read_text())
    sys.exit(int(os.environ.get('INSTALL_CURL_EXIT','0')))
elif name=='sudo':
    if args!=['-v']:os.execvp(args[0],args)
elif name=='python3.12':
    assert args[:2]==['-m','venv'];dest=Path(args[2]);(dest/'bin').mkdir(parents=True)
    (dest/'bin/python').symlink_to(Path(__file__).resolve())
    (dest/'bin/activate').write_text('export VIRTUAL_ENV='+shlex.quote(str(dest.absolute()))+'\\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\\n')
elif name=='python':
    assert args[:3]==['-m','pip','install']
    tool=Path(sys.argv[0]).with_name('uv');tool.unlink(missing_ok=True)
    tool.symlink_to(Path(__file__).resolve())
elif name=='uv':
    if args[0]=='venv':
        dest=Path(args[-1]);(dest/'bin').mkdir(parents=True)
        (dest/'bin/python').symlink_to(Path(__file__).resolve())
        (dest/'bin/activate').write_text('export VIRTUAL_ENV='+shlex.quote(str(dest.absolute()))+'\\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\\n')
    elif args[0]=='build':
        dest=Path(args[args.index('--out-dir')+1]);dest.mkdir(parents=True,exist_ok=True)
        (dest/'winding_field_cli-0.1.0-py3-none-any.whl').write_bytes(b'fixture wheel')
    elif args[:2]==['pip','install'] and (args[-1]=='.' or args[-1].endswith('.whl')):
        target=Path(args[args.index('--python')+1]) if '--python' in args else Path(sys.argv[0]).with_name('python')
        entry=target.with_name('winding-field');entry.unlink(missing_ok=True)
        entry.symlink_to(Path(__file__).resolve())
elif name=='winding-field':assert args==['doctor']
elif name!='apt-get':raise AssertionError((name,args))
''')
    driver.chmod(0o755)
    for name in ('uname','sudo','apt-get','python3.12','curl','uv'):(bin/name).symlink_to(driver)
    env=dict(os.environ,HOME=str(home),PATH=str(bin)+os.pathsep+os.environ['PATH'],
             GIT_CONFIG_GLOBAL=str(config),GIT_CONFIG_NOSYSTEM='1',INSTALL_LOG=str(log),
             INSTALL_SCRIPT=str(SCRIPT),INSTALL_RUNTIME=str(RUNTIME))
    def run(*,content=None,**overrides):
        return subprocess.run(['bash',str(SCRIPT)] if content is None else ['bash'],
                              input=content,cwd=work,env=dict(env,**overrides),
                              capture_output=True,text=True,timeout=30)
    return run,work,home,log


def test_shared_auto_runtime_installs_specs_extras_and_wheel(installer):
    run,work,home,log=installer
    result=run(content='bash "$INSTALL_RUNTIME" auto "$PWD/cli" "$HOME/lasagna"\n')
    assert result.returncode==0,result.stderr
    calls=[json.loads(l) for l in log.read_text().splitlines()]
    installs=[c for c in calls if c[:3]==['uv','pip','install']]
    assert len(installs)==3
    cli=next(c for c in installs if '--extra' in c)
    assert cli[cli.index('--requirements')+1]==str(work/'pyproject.toml')
    assert cli.count('--extra')==2 and 'plot' in cli and 'tifxyz' in cli
    assert cli[cli.index('--torch-backend')+1]=='auto'
    assert cli[cli.index('--constraint')+1]==str(ROOT/'docker/cli.in')
    lasagna=next(c for c in installs if str(home/'lasagna/bin/python') in c)
    assert lasagna[lasagna.index('--requirements')+1]==str(ROOT/'docker/lasagna.in')
    assert all('--build-constraints' in c for c in (cli,lasagna))
    assert (work/'cli/bin/winding-field').is_file()
    assert len([c for c in calls if c[:3]==['uv','pip','check']])==2
    before=(work/'cli/bin/python').lstat().st_ino
    assert run(content='bash "$INSTALL_RUNTIME" auto "$PWD/cli" "$HOME/lasagna"\n').returncode==0
    assert (work/'cli/bin/python').lstat().st_ino==before


@pytest.mark.parametrize('backend',['cpu','cu128'])
def test_shared_locked_runtime_preserves_hashes_and_dependency_layer(installer,backend):
    run,work,home,log=installer
    result=run(content=f'bash "$INSTALL_RUNTIME" {backend} "$PWD/cli" "$HOME/lasagna" dependencies\n')
    assert result.returncode==0,result.stderr
    calls=[json.loads(l) for l in log.read_text().splitlines()]
    sync=[c for c in calls if c[:3]==['uv','pip','sync']]
    assert len(sync)==2 and not any(c[:2]==['uv','build'] for c in calls)
    for c,name in zip(sync,('cli','lasagna')):
        assert '--require-hashes' in c and '--build-constraints' in c
        assert c[-1]==str(ROOT/f'docker/{name}-{backend}.txt')
        assert c[c.index('--torch-backend')+1]==backend
    assert not (work/'cli/bin/winding-field').exists()
    before=len(calls)
    result=run(content=f'bash "$INSTALL_RUNTIME" {backend} "$PWD/cli" "$HOME/lasagna" application "$PWD/wheels"\n')
    assert result.returncode==0,result.stderr
    later=[json.loads(l) for l in log.read_text().splitlines()][before:]
    assert not any(c[:3]==['uv','pip','sync'] for c in later)
    assert not any(c[:2]==['uv','venv'] for c in later)
    assert list((work/'wheels').glob('*.whl')) and (work/'cli/bin/winding-field').exists()


@pytest.mark.parametrize('arguments',['cuda cli lasagna','auto cli lasagna wrong-phase'])
def test_shared_runtime_rejects_invalid_recipe_before_writes(installer,arguments):
    run,work,_,log=installer
    result=run(content=f'bash "$INSTALL_RUNTIME" {arguments}\n')
    assert result.returncode!=0 and 'usage' in result.stderr.lower()
    assert not log.exists() and not (work/'cli').exists()


def test_shared_runtime_preserves_broken_environment(installer):
    run,work,_,log=installer
    target=work/'cli';target.mkdir();(target/'keep').write_text('user data')
    result=run(content='bash "$INSTALL_RUNTIME" auto "$PWD/cli" "$HOME/lasagna"\n')
    assert result.returncode!=0 and 'incomplete' in result.stderr
    assert (target/'keep').read_text()=='user data' and not log.exists()


@pytest.mark.parametrize('alias',['./cli','nested/../cli','cli/'])
def test_shared_runtime_rejects_equivalent_missing_environments(installer,alias):
    run,work,_,log=installer
    result=run(content=f'bash "$INSTALL_RUNTIME" auto cli {shlex.quote(alias)} dependencies\n')
    assert result.returncode!=0 and 'separate environments' in result.stderr
    assert not log.exists() and not (work/'cli').exists()


def test_shared_runtime_rejects_missing_environments_through_parent_symlink(installer):
    run,work,_,log=installer
    (work/'alias').symlink_to(work,target_is_directory=True)
    result=run(content='bash "$INSTALL_RUNTIME" auto cli alias/cli dependencies\n')
    assert result.returncode!=0 and 'separate environments' in result.stderr
    assert not log.exists() and not (work/'cli').exists()


def test_quick_install_uses_shared_runtime_specs_and_checks(installer):
    run,work,home,log=installer
    result=run()
    assert result.returncode==0,result.stderr
    calls=[json.loads(l) for l in log.read_text().splitlines()]
    assert any(c[:2]==['uv','build'] for c in calls)
    cli=next(c for c in calls if '--extra' in c)
    assert cli[cli.index('--requirements')+1]==str(work/'winding-field-cli/pyproject.toml')
    assert 'plot' in cli and 'tifxyz' in cli
    assert len([c for c in calls if c[:3]==['uv','pip','check']])==2
    packages=[c for c in calls if c[:3]==['apt-get','install','-y']]
    assert any('libcurl4-openssl-dev' in c and 'python3.12-venv' in c for c in packages)


def test_readme_manual_install_runs_shared_runtime(installer):
    run,work,home,log=installer
    section=(ROOT/'README.md').read_text().split('### 1B. Manual install',1)[1].split('</details>',1)[0]
    blocks=re.findall(r'```bash\n(.*?)```',section,re.S)
    result=run(content='set -eu\n'+'\n'.join(blocks[:2]))
    assert result.returncode==0,result.stderr
    assert (work/'winding-field-cli/.venv/bin/winding-field').exists()
    assert (home/'.cache/winding-field/lasagna-venv/bin/python').exists()
    calls=[json.loads(l) for l in log.read_text().splitlines()]
    assert any(c[:2]==['uv','build'] for c in calls)
    assert len([c for c in calls if c[:3]==['uv','pip','check']])==2


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


def test_stdin_execution_prints_activation_instructions(installer):
    run,_,_,_=installer
    result=run(content=SCRIPT.read_text())
    assert result.returncode==0,result.stderr
    assert 'source ' in result.stdout and '.venv/bin/activate' in result.stdout
    assert 'activated in this shell' not in result.stdout


def test_stdin_execution_preserves_dependency_failure_status(installer):
    run,work,_,_=installer
    result=run(content=SCRIPT.read_text(),INSTALL_FAIL='apt-get')
    assert result.returncode==42,result.stderr
    assert not (work/'winding-field-cli').exists()


def test_truncated_download_cannot_start_setup(installer):
    run,work,_,log=installer
    lines=SCRIPT.read_text().splitlines()
    result=run(content='\n'.join(lines[:len(lines)//2]))
    assert result.returncode!=0
    assert not log.exists() and not (work/'winding-field-cli').exists()


def test_sourced_install_activates_caller_without_changing_shell_settings(installer):
    run,work,_,_=installer
    command=next(line for line in (ROOT/'README.md').read_text().splitlines()
                 if line.startswith('source <('))
    result=run(content=f'''
set -eu
set -o pipefail
flags=$-
original_directory=$PWD
main() {{ printf 'caller function'; }}
{command}
[[ "$PWD" == "$original_directory" ]]
[[ "$VIRTUAL_ENV" == "$PWD/winding-field-cli/.venv" ]]
[[ "$(command -v winding-field)" == "$VIRTUAL_ENV/bin/winding-field" ]]
[[ "$-" == "$flags" ]]
[[ "$(set -o | grep '^pipefail')" == *on ]]
[[ "$(main)" == 'caller function' ]]
winding-field doctor
{command}
[[ "$VIRTUAL_ENV" == "$PWD/winding-field-cli/.venv" ]]
printf 'CALLER_ACTIVATED\\n'
''')
    assert result.returncode==0,result.stderr
    assert 'CALLER_ACTIVATED' in result.stdout
    assert (work/'winding-field-cli/.venv/bin/activate').is_file()


def test_sourced_install_does_not_enable_strict_options_in_caller(installer):
    run,_,_,_=installer
    result=run(content=f'''
set +e +u
set +o pipefail
flags=$-
source <(cat {shlex.quote(str(SCRIPT))})
status=$?
[[ "$status" == 0 && "$-" == "$flags" ]] || exit 90
[[ "$(set -o | grep '^pipefail')" == *off ]] || exit 91
[[ "$VIRTUAL_ENV" == "$PWD/winding-field-cli/.venv" ]] || exit 92
''')
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('failure',[dict(INSTALL_FAIL='curl'),dict(INSTALL_CURL_EXIT='18')])
def test_readme_quick_install_returns_download_failure_without_setup(installer,failure):
    run,work,_,log=installer
    command=next(line for line in (ROOT/'README.md').read_text().splitlines()
                 if line.startswith('source <('))
    result=run(content=f'''
export VIRTUAL_ENV=existing-caller-environment
original_path=$PATH
if {command}; then
    exit 99
fi
[[ "$VIRTUAL_ENV" == existing-caller-environment ]]
[[ "$PATH" == "$original_path" ]]
''',**failure)
    assert result.returncode==0,result.stderr
    assert not (work/'winding-field-cli').exists()
    assert not any(json.loads(line)[0]=='apt-get' for line in log.read_text().splitlines())


@pytest.mark.parametrize('failure,status',[
    (dict(INSTALL_FAIL='apt-get'),42),
    (dict(INSTALL_FAIL_STAGE='cli'),43),
    (dict(INSTALL_FAIL_STAGE='lasagna'),43),
    (dict(INSTALL_FAIL_STAGE='build'),43),
    (dict(INSTALL_FAIL_STAGE='check'),43),
    (dict(INSTALL_FAIL_STAGE='missing-helper'),127),
    (dict(INSTALL_OS='Darwin'),1),
])
def test_sourced_install_returns_failures_without_exiting_or_changing_caller(installer,failure,status):
    run,work,_,_=installer
    if failure.get('INSTALL_FAIL_STAGE')=='missing-helper':
        first=run();assert first.returncode==0,first.stderr
        (work/'winding-field-cli/tools/install-runtime.sh').unlink()
    result=run(content=f'''
set -eu
set -o pipefail
flags=$-
original_directory=$PWD
original_path=$PATH
export VIRTUAL_ENV=existing-caller-environment
if source <(cat {shlex.quote(str(SCRIPT))}); then
    exit 99
else
    status=$?
fi
[[ "$status" == {status} ]]
[[ "$PWD" == "$original_directory" ]]
[[ "$PATH" == "$original_path" ]]
[[ "$VIRTUAL_ENV" == existing-caller-environment ]]
[[ "$-" == "$flags" ]]
[[ "$(set -o | grep '^pipefail')" == *on ]]
printf 'CALLER_ALIVE\\n'
''',**failure)
    assert result.returncode==0,result.stderr
    assert 'CALLER_ALIVE' in result.stdout
    assert 'activated in this shell' not in result.stdout
