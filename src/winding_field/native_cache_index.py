"""Use the reader's SQLite library for its WAL, including standalone Python builds.

Two SQLite implementations in one process do not share their POSIX-lock bookkeeping.
Opening the native WAL through Python's bundled SQLite can invalidate live readers.
"""

from contextlib import closing, contextmanager
import ctypes as c
from functools import lru_cache
import sqlite3
import sys


@lru_cache(maxsize=None)
def _library(path):
    lib = c.CDLL(path)
    signatures = {
        'open_v2': (c.c_int, [c.c_char_p,c.POINTER(c.c_void_p),c.c_int,c.c_char_p]),
        'close': (c.c_int, [c.c_void_p]),
        'busy_timeout': (c.c_int, [c.c_void_p,c.c_int]),
        'prepare_v2': (c.c_int, [c.c_void_p,c.c_char_p,c.c_int,c.POINTER(c.c_void_p),c.POINTER(c.c_char_p)]),
        'bind_text': (c.c_int, [c.c_void_p,c.c_int,c.c_char_p,c.c_int,c.c_void_p]),
        'bind_int64': (c.c_int, [c.c_void_p,c.c_int,c.c_int64]),
        'bind_null': (c.c_int, [c.c_void_p,c.c_int]),
        'step': (c.c_int, [c.c_void_p]),
        'column_count': (c.c_int, [c.c_void_p]),
        'column_type': (c.c_int, [c.c_void_p,c.c_int]),
        'column_int64': (c.c_int64, [c.c_void_p,c.c_int]),
        'column_text': (c.c_char_p, [c.c_void_p,c.c_int]),
        'finalize': (c.c_int, [c.c_void_p]),
        'errmsg': (c.c_char_p, [c.c_void_p]),
        'get_autocommit': (c.c_int, [c.c_void_p]),
    }
    for name,(result,args) in signatures.items():
        function = getattr(lib,'sqlite3_'+name)
        function.restype,function.argtypes = result,args
    return lib


class _Rows(list):
    def fetchone(self):
        return self[0] if self else None

    def fetchall(self):
        return self


class _Connection:
    def __init__(self, library, path):
        self.lib = _library(library)
        self.db = c.c_void_p()
        code = self.lib.sqlite3_open_v2(str(path).encode(),c.byref(self.db),2,None)
        if code:
            message = self.lib.sqlite3_errmsg(self.db).decode()
            self.lib.sqlite3_close(self.db)
            raise sqlite3.OperationalError(message)
        self.lib.sqlite3_busy_timeout(self.db,120000)

    def _check(self, code):
        if code:
            raise sqlite3.OperationalError(self.lib.sqlite3_errmsg(self.db).decode())

    def execute(self, sql, parameters=()):
        statement = c.c_void_p()
        encoded = sql.encode()
        try:
            self._check(self.lib.sqlite3_prepare_v2(self.db,encoded,len(encoded),c.byref(statement),None))
            for index,value in enumerate(parameters,1):
                if value is None:
                    code = self.lib.sqlite3_bind_null(statement,index)
                elif isinstance(value,int):
                    code = self.lib.sqlite3_bind_int64(statement,index,value)
                else:
                    raw = str(value).encode()
                    code = self.lib.sqlite3_bind_text(statement,index,raw,len(raw),c.c_void_p(-1))
                self._check(code)
            rows = _Rows()
            while (code := self.lib.sqlite3_step(statement)) == 100:
                row = []
                for index in range(self.lib.sqlite3_column_count(statement)):
                    kind = self.lib.sqlite3_column_type(statement,index)
                    row.append(None if kind==5 else self.lib.sqlite3_column_int64(statement,index) if kind==1
                               else self.lib.sqlite3_column_text(statement,index).decode())
                rows.append(tuple(row))
            if code != 101:
                self._check(code)
            return rows
        finally:
            self.lib.sqlite3_finalize(statement)


@contextmanager
def cache_index(path):
    module = sys.modules.get('vc_reader._reader')
    if module is None:
        with closing(sqlite3.connect(path,timeout=120)) as db, db:
            yield db
        return
    db = _Connection(module.__file__,path)
    try:
        yield db
        if not db.lib.sqlite3_get_autocommit(db.db):
            db.execute('COMMIT')
    except BaseException:
        if not db.lib.sqlite3_get_autocommit(db.db):
            db.execute('ROLLBACK')
        raise
    finally:
        db.lib.sqlite3_close(db.db)
