"""Durable bounded units of native surface work; incomplete files are not products."""
import fcntl
from contextlib import closing
import io
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from .util import atomic_bytes,atomic_json,file_sha

SCRATCH_LIMIT=16<<30


def _scratch(name):
    return not name.startswith('meshes/') and '.npz' in name

class Paused(Exception):
    pass


class Journal:
    def __init__(self,root,identity,stop=None,seconds=1800,reserve_bytes=0):
        self.root=Path(root).absolute()
        if any(p.is_symlink() for p in (self.root,*self.root.parents)):
            raise ValueError('symlink in native output path')
        if not math.isfinite(seconds) or seconds<=0:
            raise ValueError('positive time allowance required')
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'.lock').open('a')
        self.db=None; self.stop=stop; self.seconds=seconds; self.reserve=reserve_bytes
        self.started=time.monotonic(); self.last_status=0.; self.verified={}
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            path=self.root/'IDENTITY.json'
            if path.exists():
                if json.loads(path.read_bytes())!=identity:
                    raise ValueError('native output identity changed')
            else:
                if (self.root/'work.sqlite').exists():
                    raise ValueError('native database has no identity')
                atomic_json(path,identity)
            self.identity=identity
            self.db=sqlite3.connect(self.root/'work.sqlite')
            self.db.executescript('PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA cache_size=-16384;'
                'CREATE TABLE IF NOT EXISTS files(name TEXT PRIMARY KEY,sha TEXT,bytes INTEGER);'
                'CREATE TABLE IF NOT EXISTS state(name TEXT PRIMARY KEY,value TEXT);')
            self.prior=float(self.get('elapsed_s',0.))
            self._phase='validation'; self.phase_started=self.started
            self.phase_times=self.get('timings_s',{})
            self.payload_bytes=0; committed=set()
            for name,_,_ in self.db.execute('SELECT name,sha,bytes FROM files'):
                self.validate_file(name)
                committed.add(name)
                if _scratch(name):
                    self.payload_bytes+=self.path(name).stat().st_size
            self.orphans={}
            for folder in (self.root/'charts',self.root/'uv',*self.root.glob('component-*/tiles')):
                if folder.exists():
                    self.path(str(folder.relative_to(self.root)))
                    for path in folder.rglob('*'):
                        name=str(path.relative_to(self.root))
                        if path.is_file() and name not in committed:
                            self.orphans[name]=self.path(name).stat().st_size
            self.other_scratch=0; self.last_space=0.
        except BaseException:
            if self.db is not None:
                self.db.close(); self.db=None
            self.lock.close()
            raise

    def path(self,name):
        if not isinstance(name,str) or not name or '\\' in name or any(p in ('','.','..') for p in name.split('/')):
            raise ValueError('unsafe native payload path')
        path=self.root/name
        if path.is_absolute() and not path.is_relative_to(self.root):
            raise ValueError('native payload escapes output')
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('symlink in native payload path')
        return path

    def get(self,name,default=None):
        row=self.db.execute('SELECT value FROM state WHERE name=?',(name,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set(self,name,value):
        self.db.execute('INSERT OR REPLACE INTO state VALUES (?,?)',(name,json.dumps(value,allow_nan=False)))

    @property
    def elapsed(self):
        return self.prior+time.monotonic()-self.started

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self,value):
        if value!=self._phase:
            now=time.monotonic()
            self.phase_times[self._phase]=self.phase_times.get(self._phase,0.)+now-self.phase_started
            self._phase=value; self.phase_started=now

    def timings(self):
        result=dict(self.phase_times)
        result[self.phase]=result.get(self.phase,0.)+time.monotonic()-self.phase_started
        return result

    def check(self,pending_bytes=0):
        if self.stop and self.stop() or (self.root/'STOP').exists():
            raise Paused('STOP')
        if self.elapsed>=self.seconds:
            raise Paused('TIME_LIMIT')
        if shutil.disk_usage(self.root).free<self.reserve+pending_bytes:
            raise Paused('DISK_HEADROOM')
        now=time.monotonic()
        if now-self.last_space>1:
            self.last_space=now; self.other_scratch=0
            if (self.root.parent/'NATIVE.json').is_file():
                for path in self.root.parent.glob('level-*/work.sqlite'):
                    if path.parent!=self.root:
                        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
                            row=db.execute("SELECT value FROM state WHERE name='scratch_bytes'").fetchone()
                            self.other_scratch+=0 if row is None else int(json.loads(row[0]))
        scratch=self.scratch_bytes()
        if scratch+self.other_scratch+pending_bytes>SCRATCH_LIMIT:
            raise Paused('SCRATCH_LIMIT')
        if now-self.last_status>.5:
            self.last_status=now
            with self.db:
                self.set('elapsed_s',self.elapsed)
                self.set('scratch_bytes',scratch)
                self.set('timings_s',self.timings())
            atomic_json(self.root/'STATUS.json',dict(status='RUNNING',phase=self.phase,elapsed_s=self.elapsed,
                worker_pid=os.getpid(),updated_epoch=time.time(),source_faces=self.get('source_faces'),
                face_cursor=self.get('face_cursor',0),mapped_area=self.get('mapped_area',0.),
                source_area=self.get('source_area'),render_tiles=self.get('render_tiles',0),scratch_bytes=scratch))

    def scratch_bytes(self):
        paths=[self.root/'work.sqlite',self.root/'work.sqlite-journal',self.root/'meshes/mesh.sqlite-journal',
               *(self.root/'meshes').glob('.*.pending'),*self.root.glob('component-*/.tifxyz.pending/*')]
        return self.payload_bytes+sum(self.orphans.values())+sum(p.stat().st_size for p in paths if p.is_file())

    def checkpoint(self,phase,**values):
        self.phase=phase
        with self.db:
            self.set('phase',phase)
            for name,value in values.items():
                self.set(name,value)
            self.set('elapsed_s',self.elapsed)
        self.check()

    def validate_file(self,name):
        path=self.path(name)
        row=self.db.execute('SELECT sha,bytes FROM files WHERE name=?',(name,)).fetchone()
        if row is None:
            raise ValueError('uncommitted native payload: '+name)
        stat=path.stat()
        signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if self.verified.get(name)!=signature:
            if stat.st_size!=row[1] or file_sha(path)!=row[0]:
                raise ValueError('committed native payload changed: '+name)
            self.verified[name]=signature
        return path

    def register(self,name):
        path=self.path(name)
        if self.db.execute('SELECT 1 FROM files WHERE name=?',(name,)).fetchone():
            self.validate_file(name)
            return
        if not path.is_file():
            raise ValueError('missing native payload: '+name)
        with self.db:
            self.db.execute('INSERT INTO files VALUES (?,?,?)',(name,file_sha(path),path.stat().st_size))
            self.set('elapsed_s',self.elapsed)
        if _scratch(name):
            self.payload_bytes+=path.stat().st_size
            self.orphans.pop(name,None)

    def save_arrays(self,name,**arrays):
        self.check()
        if any(np.asarray(a).dtype.kind not in 'buif' for a in arrays.values()):
            raise ValueError('numeric native payload required')
        if self.db.execute('SELECT 1 FROM files WHERE name=?',(name,)).fetchone():
            old=self.load_arrays(name)
            if old.keys()!=arrays.keys() or any(not np.array_equal(old[k],a,equal_nan=True) for k,a in arrays.items()):
                raise ValueError('immutable native payload changed: '+name)
            return
        size=sum(np.asarray(a).nbytes for a in arrays.values())
        if size>64<<20:
            raise ValueError('native unit exceeds64MiB; split the work')
        self.check(size*2+(1<<20))
        buffer=io.BytesIO(); np.savez_compressed(buffer,**arrays)
        atomic_bytes(self.path(name),buffer.getvalue())
        self.register(name)

    def load_arrays(self,name):
        path=self.validate_file(name)
        with np.load(path,allow_pickle=False) as data:
            return {k:data[k] for k in data.files}

    def close(self):
        if self.db is not None:
            with self.db:
                self.set('elapsed_s',self.elapsed)
                self.set('scratch_bytes',self.scratch_bytes())
                self.set('timings_s',self.timings())
            self.db.close(); self.db=None
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self,*unused):
        self.close()
