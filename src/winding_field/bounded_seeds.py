"""External regional Qual seed ordering on the global half-resolution lattice."""

import itertools
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np


class SeedPreparationPaused(Exception):
    """Regional preparation stopped, rather than exhausted its seed inventory."""


def seeds(fields,family,scratch,*,quota_bytes,region_size=1024,chunk=32,stop=None):
    """Yield stable sequence/ZYX/priority; ties use ascending global coordinates.

    The pinned tracer groups regions in ZYX order, then descending priority and
    centrality. Its unstable centrality ties are deliberately made reproducible.
    """
    if (family not in (1,2) or type(chunk) is not int or chunk<2 or chunk%2
            or type(region_size) is not int or region_size<chunk or region_size%chunk
            or type(quota_bytes) is not int or quota_bytes<=0):
        raise ValueError('invalid global seed enumeration configuration')
    sequence=0
    for origin in itertools.product(*(range(0,n,region_size) for n in fields.shape)):
        if stop and stop():raise SeedPreparationPaused()
        # Interrupted attempts may leave spools. Charge them; never delete an
        # unverified directory just because its name resembles our scratch.
        abandoned=sum(p.stat().st_size for folder in Path(scratch).glob('qual-seeds-*')
                      for p in folder.rglob('*') if p.is_file())
        end=np.minimum(np.asarray(origin)+region_size,fields.shape)
        with TemporaryDirectory(prefix='qual-seeds-',dir=scratch) as directory:
            db=sqlite3.connect(Path(directory)/'seeds.sqlite')
            try:
                db.execute('PRAGMA cache_size=-2048')
                db.execute('''CREATE TABLE seeds(priority INTEGER,centrality INTEGER,
                    z INTEGER,y INTEGER,x INTEGER,PRIMARY KEY(priority DESC,centrality DESC,z,y,x)) WITHOUT ROWID''')
                for start in itertools.product(*(range(a,int(b),chunk) for a,b in zip(origin,end))):
                    if stop and stop():raise SeedPreparationPaused()
                    lo=np.asarray(start);hi=np.minimum(lo+chunk,end)
                    cent,ridge=fields.centrality(lo,hi,family)
                    if cent.shape!=tuple(hi-lo) or ridge.shape!=cent.shape:
                        raise ValueError('seed stencil shape mismatch')
                    positions=np.argwhere(ridge>0)
                    size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
                    # Conservative per-row growth and rollback space, independent of compression.
                    if abandoned+2*(size+len(positions)*128+65536)>quota_bytes:
                        raise OSError('regional seed spool quota exhausted; no seeds omitted')
                    with db:
                        db.executemany('INSERT INTO seeds VALUES (?,?,?,?,?)',
                            ((int(ridge[tuple(q)]),int(cent[tuple(q)]),*map(int,q+lo)) for q in positions))
                for priority,z,y,x in db.execute('SELECT priority,z,y,x FROM seeds ORDER BY priority DESC,centrality DESC,z,y,x'):
                    if stop and stop():raise SeedPreparationPaused()
                    yield sequence,(z,y,x),priority
                    sequence+=1
            finally:
                db.close()
