"""Opt-in synthetic solver evidence. Run each source revision in a fresh process.

Timing runs are uninstrumented. --profile is a separate, synchronizing diagnostic
run with exclusive phase times, NVTX ranges and a PyTorch CPU/CUDA trace.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import inspect
import io
import itertools
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path,help='core checkout; otherwise use installed core')
    parser.add_argument('--cli',type=Path,help='CLI checkout; otherwise use installed CLI')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['solve','blocks'],default='solve')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--size',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--cache-bytes',type=int,default=0)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if args.size<4 or args.repeats<1 or args.cache_bytes<0:
        parser.error('size>=4, repeats>=1 and cache-bytes>=0 required')
    for root in (args.cli,args.core):
        if root is not None:sys.path.insert(0,str(root.resolve()/'src'))
    import numpy as np
    import torch
    import field_solver
    import block_solver
    import constraint_templates
    import field_constraints
    import field_store
    from winding_field.bounded_constraints import PairCatalog,build
    from winding_field.global_residual import verify

    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    cuda=torch.device(args.device).type=='cuda'
    def sync():
        if cuda:torch.cuda.synchronize(args.device)

    phases={}; nesting=[]
    @contextmanager
    def phase(name):
        sync(); start=time.perf_counter(); nesting.append([start,0.])
        if cuda:torch.cuda.nvtx.range_push(name)
        with torch.profiler.record_function(name):
            try:yield
            finally:
                sync(); elapsed=time.perf_counter()-start
                _,children=nesting.pop()
                if nesting:nesting[-1][1]+=elapsed
                value=phases.setdefault(name,dict(seconds=0.,calls=0))
                value['seconds']+=elapsed-children;value['calls']+=1
                if cuda:torch.cuda.nvtx.range_pop()

    def instrument(owner,name,label,stack):
        original=getattr(owner,name)
        def wrapped(*a,**kw):
            with phase(label):return original(*a,**kw)
        stack.enter_context(patch.object(owner,name,wrapped))

    rng=np.random.default_rng(423);n=args.size;shape=(n,n,n)
    target=rng.normal(0,.1,(*shape,3)).astype('f4')
    weights=rng.uniform(.5,2,shape).astype('f4')
    observed=np.ones(shape,bool)
    # Physical, nonlocal fibre pairs with multiplicity, plus nonuniform observations.
    starts=rng.uniform(0,n-1,(n*4,3));ends=rng.uniform(0,n-1,(n*4,3))
    stencils=constraint_templates.StencilConstraints.pairs(starts,ends,[0]*3,[1]*3,shape,weight=2.)
    rows=stencils.restrict([0]*3,shape,lambda nodes:np.zeros(nodes.shape))
    prepared=field_solver.PreparedField() if args.cache_bytes else None
    sources={}
    for module in (field_solver,block_solver,constraint_templates,field_constraints,field_store,
                   sys.modules[PairCatalog.__module__],sys.modules[verify.__module__]):
        path=Path(module.__file__)
        sources[module.__name__]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
                threads=torch.get_num_threads(),sources=sources,measurements=[],phases=phases,
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if cuda:
        report['gpu']=torch.cuda.get_device_name(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    fields={}
    if args.case=='blocks':
        source=args.output/'fibres.sqlite'
        with sqlite3.connect(source) as db:
            db.execute('CREATE TABLE curves(id INTEGER PRIMARY KEY,points BLOB)')
            for i,(a,b) in enumerate(zip(starts,ends)):
                stream=io.BytesIO();np.save(stream,np.array([a,b]))
                db.execute('INSERT INTO curves VALUES (?,?)',(i,stream.getvalue()))
        grid=dict(shape_zyx=list(shape),origin_xyz=[0.]*3,spacing_xyz=[1.]*3)
        build(args.output/'pairs',source,hashlib.sha256(source.read_bytes()).hexdigest(),grid,
              weight=2.,quota_bytes=128<<20)

    with ExitStack() as stack:
        profiler=None
        if args.profile:
            for owner,name,label in [
                (field_solver,'_array','input_validation_transfer'),
                (field_solver,'_norm','reductions'),(torch,'dot','reductions'),
                (field_solver._FieldOperator,'__init__','gradient_setup'),
                (field_solver._FieldOperator,'matvec','gradient_operator'),
                (field_solver._FieldOperator,'rhs','gradient_rhs'),
                (field_constraints._QuadraticOperator,'__init__','constraint_setup_transfer'),
                (field_constraints._QuadraticOperator,'matvec','constraint_operator'),
                (field_constraints._QuadraticOperator,'rhs','constraint_rhs'),
                (PairCatalog,'__call__','restriction'),(field_store.BlockStore,'commit','checkpoint'),
                (block_solver,'_numpy','output_transfer'),
            ]:instrument(owner,name,label,stack)
            activities=[torch.profiler.ProfilerActivity.CPU]
            if cuda:activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler=stack.enter_context(torch.profiler.profile(activities=activities))
        for repetition in range(args.repeats):
            sync();start=time.perf_counter()
            if args.case=='solve':
                field,receipt=field_solver.solve_field(target,weights,constraints=rows,prepared=prepared,
                    device=args.device,dtype='float64',rtol=1e-8,maxiter=2000)
                sync();elapsed=time.perf_counter()-start
                if not receipt['converged']:raise RuntimeError(receipt)
                result=dict(solver=receipt,constraint_energy=float(np.sum(rows.weights*rows.residual(field.ravel())**2)))
            else:
                root=args.output/f'blocks-{repetition}';chunk=max(2,n//2)
                def load(lo,hi):
                    s=tuple(slice(a,b) for a,b in zip(lo,hi))
                    return target[s],weights[s],observed[s],{'synthetic':'a'*64}
                cache_options={}
                if 'prepared_cache_bytes' in inspect.signature(block_solver.solve_blocks).parameters:
                    cache_options['prepared_cache_bytes']=args.cache_bytes
                provider_options={}
                if 'cache_bytes' in inspect.signature(PairCatalog).parameters:
                    provider_options['cache_bytes']=args.cache_bytes
                solves=[]
                original_solve=block_solver.solve_field
                def counted_solve(*a,**kw):
                    field,info=original_solve(*a,**kw)
                    solves.append(info)
                    return field,info
                with PairCatalog(args.output/'pairs',grid,**provider_options) as provider:
                    with patch.object(block_solver,'solve_field',counted_solve):
                        receipt=block_solver.solve_blocks(root,shape,chunk,{'synthetic':423},load,
                            origin_xyz=[0]*3,spacing_xyz=[1]*3,
                            order=list(itertools.product(*(range((d-1)//chunk+1) for d in shape))),
                            constraints=provider,halo=1,sweeps=3,rtol=1e-8,maxiter=2000,device=args.device,
                            **cache_options)
                    sync();elapsed=time.perf_counter()-start
                    cp=json.loads((root/'CHECKPOINT.json').read_text())
                    with field_store.BlockStore(root,shape,chunk,cp['identity'],resume=True) as store:
                        field=store.read([0]*3,shape)
                        validation=verify(shape,chunk,[1]*3,store.read,load,constraints=provider,
                                          gauge_offset=store._load((0,0,0))[1])
                    result=dict(solver=cp['progress']['solver'],block_status=receipt,global_residual=validation,
                                local_solves=solves)
                    if hasattr(provider,'stats'):result['constraint_cache_stats']=dict(provider.stats)
            result.update(seconds=elapsed,temperature='cold' if repetition==0 else 'warm')
            report['measurements'].append(result);fields[f'field_{repetition}']=field.copy()
            if prepared is not None and hasattr(prepared,'retained_bytes'):
                report['prepared_retained_bytes']=prepared.retained_bytes
    if profiler is not None:
        profiler.export_chrome_trace(str(args.output/'trace.json'))
        report['operators']=profiler.key_averages().table(sort_by='self_cpu_time_total',row_limit=30)
    if cuda:
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
        report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(args.device)
    np.savez(args.output/'fields.npz',**fields)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),seconds=[m['seconds'] for m in report['measurements']])))


if __name__=='__main__':main()
