"""Disk-backed indexed surfaces: disjoint cell tiles, global edges and atomic cursors."""

import fcntl
import hashlib
import itertools
import json
import marshal
import math
import os
import sqlite3
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

import field_store
from field_runtime import check_resources
from field_store import atomic_json


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _levels(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("nonempty raw field levels required")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("levels must be finite real numbers")
    result = list(map(float, values))
    if len(set(result)) != len(result):
        raise ValueError("duplicate levels")
    return result


def _signature(path):
    if Path(path).is_symlink():
        raise ValueError("mesh member cannot be a symlink")
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _rows_sha(db, query, args=()):
    digest = hashlib.sha256()
    for row in db.execute(query, args):
        digest.update(json.dumps(row, separators=(",", ":"), allow_nan=False).encode() + b"\n")
    return digest.hexdigest()


def _derived_sha(db, summary):
    parts = [
        _rows_sha(db, "SELECT id,parent FROM faces ORDER BY id"),
        _rows_sha(db, "SELECT * FROM edges ORDER BY level,a,b"),
        _rows_sha(db, "SELECT * FROM components ORDER BY level,root"),
    ]
    return hashlib.sha256(json.dumps([parts, summary], sort_keys=True, allow_nan=False).encode()).hexdigest()


def _manifest(metadata, identity, summary, total, database_sha):
    return dict(
        schema="winding-mesh-v1",
        status="MESH_READY",
        metadata=metadata,
        levels=summary,
        next_tile=total,
        total_tiles=total,
        database_sha256=database_sha,
        physical_acceptance=False,
        self_intersection="UNMEASURED",
        identity=identity,
    )


class MeshDB:
    """One SQLite writer; metadata, all geometry and tile cursor commit together."""

    def __init__(
        self, path: str | Path, metadata: dict, identity: dict, *, resume: bool = False, reserve_bytes: int = 0
    ):
        self.root = Path(path)
        self.db = None
        self.lock = None
        self.reserve_bytes = _integer(reserve_bytes, "reserve_bytes", 0)
        if type(resume) is not bool or self.root.is_symlink():
            raise ValueError("invalid mesh resume/path")
        self.binding = dict(schema="winding-mesh-db-v1", metadata=metadata, identity=identity)
        raw = json.dumps(self.binding, sort_keys=True, allow_nan=False)
        check_resources(self.root.parent, reserve_bytes=self.reserve_bytes + 1024**2)
        if not resume:
            self.root.mkdir(exist_ok=False)
            atomic_json(self.root / "RUN.json", self.binding)
        elif json.loads((self.root / "RUN.json").read_bytes()) != self.binding:
            raise ValueError("mesh source/config identity differs")
        self.run_sha = sha(self.root / "RUN.json")
        try:
            self.lock = (self.root / ".mesh.lock").open("a")
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            dbpath = self.root / "mesh.sqlite"
            if dbpath.is_symlink() or (resume and not dbpath.is_file()):
                raise ValueError("invalid mesh database path")
            self.db = sqlite3.connect(dbpath)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA cache_size=-16384")
            self.db.execute("PRAGMA temp_store=FILE")
            if resume:
                if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("mesh database integrity failure")
                if self.get("binding") != json.loads(raw):
                    raise ValueError("mesh database identity differs")
                self.validate_tiles()
                if self.get("complete"):
                    if self.get("derived_sha") != _derived_sha(self.db, self.get("summary")):
                        raise ValueError("mesh derived state checksum differs")
                    manifest = self.root / "MESH.json"
                    if manifest.exists() and json.loads(manifest.read_bytes()) != _manifest(
                        metadata, identity, self.get("summary"), identity["total_tiles"], sha(dbpath)
                    ):
                        raise ValueError("completed mesh database/manifest changed")
                else:
                    with self.db:
                        self.rebuild_topology()
            else:
                self.db.executescript("""
                    CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                    CREATE TABLE levels(id INTEGER PRIMARY KEY,value REAL UNIQUE NOT NULL);
                    CREATE TABLE vertices(id INTEGER PRIMARY KEY,level INTEGER NOT NULL,key TEXT NOT NULL,
                      x REAL NOT NULL,y REAL NOT NULL,z REAL NOT NULL,local INTEGER NOT NULL,
                      UNIQUE(level,key),UNIQUE(level,local));
                    CREATE TABLE faces(id INTEGER PRIMARY KEY,level INTEGER NOT NULL,a INTEGER NOT NULL,
                      b INTEGER NOT NULL,c INTEGER NOT NULL,parent INTEGER,area REAL NOT NULL,key TEXT NOT NULL);
                    CREATE INDEX face_level ON faces(level,id);
                    CREATE INDEX face_key ON faces(level,key);
                    CREATE TABLE edges(level INTEGER,a INTEGER,b INTEGER,face INTEGER,count INTEGER,sign INTEGER,
                      PRIMARY KEY(level,a,b));
                    CREATE TABLE components(level INTEGER,root INTEGER,faces INTEGER,area REAL,
                      PRIMARY KEY(level,root));
                    CREATE TABLE tiles(id INTEGER PRIMARY KEY,vertex_last INTEGER,face_last INTEGER,
                      vertex_sha TEXT,face_sha TEXT);
                """)
                self.set("binding", self.binding)
                self.set("next_tile", 0)
                self.set("complete", False)
                self.db.executemany("INSERT INTO levels VALUES(?,?)", enumerate(identity.get("levels", [0.0])))
                self.db.commit()
            s = dbpath.stat()
            self.db_inode = s.st_dev, s.st_ino
            s = os.fstat(self.lock.fileno())
            self.lock_inode = s.st_dev, s.st_ino
        except BaseException:
            self.close()
            raise

    def get(self, key):
        return json.loads(self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])

    def set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value, allow_nan=False)))

    def check(self):
        for path, expected in ((self.root / "mesh.sqlite", self.db_inode), (self.root / ".mesh.lock", self.lock_inode)):
            if path.is_symlink() or (path.stat().st_dev, path.stat().st_ino) != expected:
                raise ValueError("mesh writer path identity changed")
        if sha(self.root / "RUN.json") != self.run_sha:
            raise ValueError("mesh RUN identity changed")

    def admit(self):
        check_resources(self.root, reserve_bytes=self.reserve_bytes + 32 * 1024**2)

    def validate_tiles(self):
        if list(self.db.execute("SELECT id,value FROM levels ORDER BY id")) != list(
            enumerate(self.binding["identity"].get("levels", [0.0]))
        ):
            raise ValueError("mesh level inventory differs")
        count = 0
        previous_v = previous_f = 0
        for index, last_v, last_f, vsha, fsha in self.db.execute("SELECT * FROM tiles ORDER BY id"):
            if index != count or last_v < previous_v or last_f < previous_f:
                raise ValueError("invalid committed tile journal")
            if (
                self.row_digest("vertices", previous_v, last_v) != vsha
                or self.row_digest("faces", previous_f, last_f) != fsha
            ):
                raise ValueError("committed mesh geometry checksum differs")
            previous_v, previous_f = last_v, last_f
            count += 1
        cursor = self.get("next_tile")
        if (
            type(cursor) is not int
            or cursor != count
            or count > self.binding["identity"]["total_tiles"]
            or self.last_id("vertices") != previous_v
            or self.last_id("faces") != previous_f
            or type(self.get("complete")) is not bool
            or (self.get("complete") and count != self.binding["identity"]["total_tiles"])
        ):
            raise ValueError("mesh cursor and committed tile inventory differ")

    def last_id(self, table):
        return self.db.execute("SELECT COALESCE(MAX(id),0) FROM " + table).fetchone()[0]

    def row_digest(self, table, first, last):
        fields = "id,level,key,x,y,z,local" if table == "vertices" else "id,level,a,b,c,area,key"
        return _rows_sha(
            self.db, "SELECT " + fields + " FROM " + table + " WHERE id>? AND id<=? ORDER BY id", (first, last)
        )

    def checkpoint_tile(self, index):
        if index != self.get("next_tile"):
            raise ValueError("nonsequential mesh tile commit")
        row = self.db.execute("SELECT vertex_last,face_last FROM tiles ORDER BY id DESC LIMIT 1").fetchone()
        first_v, first_f = row if row else (0, 0)
        last_v, last_f = self.last_id("vertices"), self.last_id("faces")
        self.db.execute(
            "INSERT INTO tiles VALUES(?,?,?,?,?)",
            (
                index,
                last_v,
                last_f,
                self.row_digest("vertices", first_v, last_v),
                self.row_digest("faces", first_f, last_f),
            ),
        )
        self.set("next_tile", index + 1)
        check_resources(self.root, reserve_bytes=self.reserve_bytes + 1024**2)

    def link_face(self, identity, level, a, b, c):
        for i, j in ((a, b), (b, c), (c, a)):
            lo, hi = sorted((i, j))
            sign = 1 if i < j else -1
            row = self.db.execute("SELECT face FROM edges WHERE level=? AND a=? AND b=?", (level, lo, hi)).fetchone()
            if row:
                r1, r2 = self.find(identity), self.find(row[0])
                self.db.execute("UPDATE faces SET parent=? WHERE id=?", (min(r1, r2), max(r1, r2)))
                self.db.execute(
                    "UPDATE edges SET count=count+1,sign=sign+? WHERE level=? AND a=? AND b=?", (sign, level, lo, hi)
                )
            else:
                self.db.execute("INSERT INTO edges VALUES(?,?,?,?,1,?)", (level, lo, hi, identity, sign))

    def rebuild_topology(self):
        self.db.execute("DELETE FROM edges")
        self.db.execute("DELETE FROM components")
        self.db.execute("UPDATE faces SET parent=id")
        for index, (identity, level, a, b, c) in enumerate(
            self.db.execute("SELECT id,level,a,b,c FROM faces ORDER BY id")
        ):
            if index % 256 == 0:
                self.admit()
            self.link_face(identity, level, a, b, c)

    def find(self, identity):
        trail = []
        while True:
            parent = self.db.execute("SELECT parent FROM faces WHERE id=?", (identity,)).fetchone()[0]
            if parent == identity:
                break
            if parent is None or parent > identity:
                raise ValueError("invalid mesh component parent")
            trail.append(identity)
            identity = parent
        self.db.executemany("UPDATE faces SET parent=? WHERE id=?", ((identity, i) for i in trail))
        return identity

    def add(self, level, keys, vertices, faces):
        vertices, faces = np.asarray(vertices), np.asarray(faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not np.isfinite(vertices).all()
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or faces.dtype.kind not in "iu"
            or len(keys) != len(vertices)
            or (len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)))
        ):
            raise ValueError("invalid mesh arrays")
        local = self.db.execute("SELECT COALESCE(MAX(local),0)+1 FROM vertices WHERE level=?", (level,)).fetchone()[0]
        ids = []
        for index, (key, point) in enumerate(zip(keys, vertices)):
            if index % 256 == 0:
                self.admit()
            found = self.db.execute("SELECT id,x,y,z FROM vertices WHERE level=? AND key=?", (level, key)).fetchone()
            if found:
                if tuple(point) != found[1:]:
                    raise ValueError("shared lattice vertex coordinates differ")
                ids.append(found[0])
            else:
                cur = self.db.execute(
                    "INSERT INTO vertices(level,key,x,y,z,local) VALUES(?,?,?,?,?,?)",
                    (level, key, *map(float, point), local),
                )
                ids.append(cur.lastrowid)
                local += 1
        for index, face in enumerate(faces):
            if index % 128 == 0:
                self.admit()
            a, b, c = (ids[int(i)] for i in face)
            tri = vertices[face]
            area = float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / 2)
            if len({a, b, c}) != 3 or area == 0:
                continue
            if not math.isfinite(area):
                raise ValueError("mesh area exceeds numerical capacity")
            identity = self.db.execute(
                "INSERT INTO faces(level,a,b,c,area,key) VALUES(?,?,?,?,?,?)",
                (level, a, b, c, area, ",".join(map(str, sorted((a, b, c))))),
            ).lastrowid
            self.db.execute("UPDATE faces SET parent=id WHERE id=?", (identity,))
            self.link_face(identity, level, a, b, c)

    def finish(self):
        self.check()
        if self.get("complete"):
            summary = self.get("summary")
            for item in summary:
                self.export(item["id"])
            return summary
        with self.db:
            self.db.execute("DELETE FROM components")
            cursor = self.db.execute("SELECT id,level,area FROM faces ORDER BY id")
            while rows := cursor.fetchmany(8192):
                for identity, level, area in rows:
                    root = self.find(identity)
                    self.db.execute(
                        "INSERT INTO components VALUES(?,?,1,?) ON CONFLICT(level,root) DO UPDATE SET "
                        "faces=faces+1,area=area+excluded.area",
                        (level, root, area),
                    )
        summary = []
        for level, value in self.db.execute("SELECT id,value FROM levels ORDER BY id"):
            n = lambda table: self.db.execute("SELECT COUNT(*) FROM " + table + " WHERE level=?", (level,)).fetchone()[
                0
            ]
            edges = self.db.execute(
                "SELECT SUM(count=1),SUM(count>2),SUM(count=2 AND sign!=0) FROM edges WHERE level=?", (level,)
            ).fetchone()
            duplicates = self.db.execute(
                "SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM faces "
                "WHERE level=? GROUP BY key HAVING COUNT(*)>1)",
                (level,),
            ).fetchone()[0]
            summary.append(
                dict(
                    id=level,
                    level=value,
                    vertices=n("vertices"),
                    faces=n("faces"),
                    components=n("components"),
                    area_world2=self.db.execute(
                        "SELECT COALESCE(SUM(area),0) FROM faces WHERE level=?", (level,)
                    ).fetchone()[0],
                    boundary_edges=edges[0] or 0,
                    nonmanifold_edges=edges[1] or 0,
                    inconsistent_edges=edges[2] or 0,
                    duplicate_faces=duplicates,
                )
            )
        with self.db:
            self.set("summary", summary)
            self.set("derived_sha", _derived_sha(self.db, summary))
            self.set("complete", True)
        for item in summary:
            self.export(item["id"])
        return summary

    def export(self, level):
        path = self.root / f"level-{level:04d}.obj"
        pending = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
        with pending.open("x") as stream:
            stream.write("# World XYZ; geometry metadata in RUN.json\n")
            for index, (x, y, z) in enumerate(
                self.db.execute("SELECT x,y,z FROM vertices WHERE level=? ORDER BY local", (level,))
            ):
                if index % 1024 == 0:
                    check_resources(self.root, reserve_bytes=self.reserve_bytes + 1024**2)
                stream.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
            query = (
                "SELECT va.local,vb.local,vc.local FROM faces f JOIN vertices va ON va.id=f.a "
                "JOIN vertices vb ON vb.id=f.b JOIN vertices vc ON vc.id=f.c WHERE f.level=? ORDER BY f.id"
            )
            for index, (a, b, c) in enumerate(self.db.execute(query, (level,))):
                if index % 1024 == 0:
                    check_resources(self.root, reserve_bytes=self.reserve_bytes + 1024**2)
                stream.write(f"f {a} {b} {c}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if path.is_symlink() or sha(path) != sha(pending):
                raise ValueError("existing OBJ differs; preserve it and use a new output")
            pending.unlink()
        else:
            pending.replace(path)

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


class _LatticePrecisionError(ValueError):
    pass


def _tile_mesh_once(values, valid, low, level, origin, spacing, flips=()):
    from skimage.measure import marching_cubes

    cells = np.ones(tuple(n - 1 for n in values.shape), bool)
    for bits in itertools.product((0, 1), repeat=3):
        cells &= valid[tuple(slice(b, n - 1 + b) for b, n in zip(bits, values.shape))]
    if not cells.any() or not valid.any() or not values[valid].min() <= level <= values[valid].max():
        return [], np.empty((0, 3)), np.empty((0, 3), np.int64)
    work = np.where(valid, values - level, 0).astype(np.float32)
    if not np.isfinite(work).all():
        raise ValueError("shifted field cannot be represented as finite float32")
    if (valid & (work == 0) & (values != level)).any():
        raise _LatticePrecisionError("level crossing underflows float32")
    if flips:
        work = np.flip(work, axis=flips)
    mask = np.zeros(values.shape, bool)
    mask[1:, 1:, 1:] = cells
    try:
        # Keep tiny faces until their edge positions have been reconstructed in float64.
        vertices, faces, _, _ = marching_cubes(work, 0.0, mask=mask, allow_degenerate=True)
    except RuntimeError as error:
        if "No surface" not in str(error):
            raise
        return [], np.empty((0, 3)), np.empty((0, 3), np.int64)
    vertices = vertices.astype(np.float64)
    for axis in flips:
        vertices[:, axis] = values.shape[axis] - 1 - vertices[:, axis]
    if len(flips) % 2:
        faces = faces[:, ::-1]
    keys, world = [], []
    for vertex in vertices:
        base = np.floor(vertex).astype(np.int64)
        frac = vertex - base
        axes = np.flatnonzero(frac)
        global_base = base + low
        position = vertex + low
        if len(axes) == 0:
            if values[tuple(base)] != level:
                raise _LatticePrecisionError("level crossing rounded to a lattice node")
            key = "n:" + ",".join(map(str, global_base))
        elif len(axes) == 1:
            axis = int(axes[0])
            other = base.copy()
            other[axis] += 1
            a, b = float(values[tuple(base)]), float(values[tuple(other)])
            position = global_base.astype(float)
            if a == level or b == level:
                endpoint = base if a == level else other
                position = (endpoint + low).astype(float)
                key = "n:" + ",".join(map(str, endpoint + low))
            else:
                position[axis] += (level - a) / (b - a)
                key = f"e:{axis}:" + ",".join(map(str, global_base))
        else:
            key = "p:" + ",".join(float(v).hex() for v in position)
        keys.append(key)
        world.append(origin + position[::-1] * spacing)
    return keys, np.asarray(world), faces[:, ::-1]


def _tile_mesh(values, valid, low, level, origin, spacing):
    try:
        return _tile_mesh_once(values, valid, low, level, origin, spacing)
    except _LatticePrecisionError:
        cells = np.asarray(values.shape) - 1
        if (cells == 1).all():
            # A crossing near zero retains precision that float32 loses near one.
            for flipped in itertools.product((False, True), repeat=3):
                axes = tuple(i for i, use in enumerate(flipped) if use)
                if not axes:
                    continue
                try:
                    return _tile_mesh_once(values, valid, low, level, origin, spacing, axes)
                except _LatticePrecisionError:
                    continue
            raise
        ranges = [[(0, n)] if n == 1 else [(0, n // 2), (n // 2, n)] for n in cells]
        keys, vertices, faces = [], [], []
        offset = 0
        for bounds in itertools.product(*ranges):
            first = np.array([a for a, b in bounds])
            crop = tuple(slice(a, b + 1) for a, b in bounds)
            k, v, f = _tile_mesh(values[crop], valid[crop], low + first, level, origin, spacing)
            keys.extend(k)
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        return keys, np.concatenate(vertices), np.concatenate(faces)


def extract_mesh(
    source,
    output: str | Path,
    levels,
    *,
    tile: int = 128,
    level_batch: int = 4,
    resume: bool = False,
    stop=None,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Read each owned cell tile once; preserve all supported components on disk."""
    tile, level_batch = _integer(tile, "tile"), _integer(level_batch, "level_batch")
    reserve_bytes = _integer(reserve_bytes, "reserve_bytes", 0)
    levels = _levels(levels)
    if type(resume) is not bool or (stop is not None and not callable(stop)) or min(source.shape) < 2:
        raise ValueError("invalid extraction callback/resume/grid")
    if tile > 256:
        raise ValueError("tile exceeds bounded256-cell extraction; use smaller tiles")
    source.validate(hashes=resume)
    grid = tuple((n - 2) // tile + 1 for n in source.shape)
    total = math.prod(grid)
    identity = dict(
        source=source.identity,
        levels=levels,
        tile=tile,
        total_tiles=total,
        algorithm="lewiner-shifted-f32-adaptive-global-edges-v2",
        numpy=np.__version__,
        skimage=version("scikit-image"),
        code={
            Path(__file__).name: sha(__file__),
            "surface_sources.py": sha(Path(__file__).with_name("surface_sources.py")),
            "field_store.py": sha(field_store.__file__),
        },
        decoder_runtime=hashlib.sha256(marshal.dumps(field_store._decode.__code__)).hexdigest(),
    )
    metadata = source.identity["geometry"]
    output = Path(output)
    check_resources(output.parent, reserve_bytes=reserve_bytes)
    with MeshDB(output, metadata, identity, resume=resume, reserve_bytes=reserve_bytes) as store:
        first = store.get("next_tile")
        if type(first) is not int or not 0 <= first <= total:
            raise ValueError("invalid mesh cursor")
        for index, key in enumerate(itertools.product(*(range(n) for n in grid))):
            if index < first:
                continue
            if (output / "STOP").exists() or (stop is not None and stop()):
                result = dict(status="CHECKPOINTED_STOP", next_tile=index, total_tiles=total, physical_acceptance=False)
                atomic_json(output / "STATUS.json", result)
                return result
            store.check()
            low = np.asarray(key) * tile
            high = np.minimum(low + tile, np.asarray(source.shape) - 1)
            check_resources(output, reserve_bytes=reserve_bytes + int(np.prod(high - low)) * 128)
            values, valid = source.read(low, high + 1)
            with store.db:
                for start in range(0, len(levels), level_batch):
                    for j in range(start, min(start + level_batch, len(levels))):
                        keys, vertices, faces = _tile_mesh(values, valid, low, levels[j], source.origin, source.spacing)
                        if len(faces):
                            store.add(j, keys, vertices, faces)
                store.checkpoint_tile(index)
            atomic_json(output / "STATUS.json", dict(status="EXTRACTING", next_tile=index + 1, total_tiles=total))
        source.validate()
        details = store.finish()
    result = _manifest(metadata, identity, details, total, sha(output / "mesh.sqlite"))
    atomic_json(output / "MESH.json", result)
    atomic_json(output / "STATUS.json", result)
    return result


def _npz_geometry(stream, max_bytes):
    with ZipFile(stream) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or sum(i.file_size for i in infos) > max_bytes:
            raise ValueError("NPZ exceeds import budget or contains duplicate members")
        result = []
        charge = 0
        for name in ("vertices_xyz.npy", "faces.npy"):
            if name not in names:
                raise ValueError("mesh NPZ requires vertices_xyz and faces")
            with archive.open(name) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("unsupported mesh NPY header")
                header = member.tell()
            if len(shape) != 2 or shape[1] != 3 or dtype.kind not in ("iu" if name == "faces.npy" else "iuf"):
                raise ValueError("invalid mesh NPZ header shape/dtype")
            size = math.prod(shape) * dtype.itemsize
            charge += size
            if charge > max_bytes or archive.getinfo(name).file_size != header + size:
                raise ValueError("mesh NPY header/data byte count exceeds budget or differs")
            result.append(field_store._array(archive.read(name), (shape,), dtype))
        return result


def import_mesh(
    descriptor: dict,
    output: str | Path,
    metadata: dict,
    *,
    resume: bool = False,
    max_bytes: int = 512 * 1024**2,
    reserve_bytes: int = 0,
) -> dict:
    """Import numeric world-XYZ NPZ or streaming triangulated OBJ; preserve source indices."""
    max_bytes = _integer(max_bytes, "max_bytes")
    path = Path(descriptor["path"])
    signature = _signature(path)
    if sha(path) != descriptor["sha256"] or _signature(path) != signature:
        raise ValueError("mesh source hash/identity differs")
    if path.suffix.lower() not in (".obj", ".npz"):
        raise ValueError("mesh source must be triangular OBJ or numeric NPZ")
    identity = dict(
        input=descriptor,
        kind="imported_world_mesh",
        total_tiles=1,
        code_sha256=sha(__file__),
        decoder_sha256=sha(field_store.__file__),
        numpy=np.__version__,
    )
    output = Path(output)
    check_resources(output.parent, reserve_bytes=reserve_bytes)
    with MeshDB(output, metadata, identity, resume=resume, reserve_bytes=reserve_bytes) as store:
        if not store.get("next_tile"):
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != descriptor["sha256"]:
                    raise ValueError("mesh source hash differs")
                stream.seek(0)
                with store.db:
                    if path.suffix.lower() == ".npz":
                        vertices, faces = _npz_geometry(stream, max_bytes)
                        if (
                            vertices.ndim != 2
                            or vertices.shape[1:] != (3,)
                            or vertices.dtype.kind not in "iuf"
                            or not np.isfinite(vertices).all()
                            or faces.ndim != 2
                            or faces.shape[1:] != (3,)
                            or faces.dtype.kind not in "iu"
                            or (len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)))
                        ):
                            raise ValueError("invalid numeric mesh arrays")
                        for start in range(0, len(vertices), 8192):
                            part = vertices[start : start + 8192]
                            store.add(
                                0, [f"input:{i}" for i in range(start, start + len(part))], part, np.empty((0, 3), int)
                            )
                        for start in range(0, len(faces), 8192):
                            vertex_ids, remap = np.unique(faces[start : start + 8192], return_inverse=True)
                            store.add(0, [f"input:{i}" for i in vertex_ids], vertices[vertex_ids], remap.reshape(-1, 3))
                    else:
                        count = 0
                        while raw := stream.readline(1024**2 + 1):
                            if len(raw) > 1024**2:
                                raise ValueError("OBJ line exceeds byte limit")
                            words = raw.decode("utf-8").split("#", 1)[0].split()
                            if not words:
                                continue
                            if words[0] == "v":
                                if len(words) != 4:
                                    raise ValueError("OBJ requires XYZ vertices")
                                point = np.array([float(v) for v in words[1:]])
                                store.add(0, [f"input:{count}"], point[None, :], np.empty((0, 3), int))
                                count += 1
                            elif words[0] == "f":
                                if len(words) != 4:
                                    raise ValueError("OBJ must be triangulated")
                                raw_ids = [int(v.split("/", 1)[0]) for v in words[1:]]
                                ids = [v - 1 if v > 0 else count + v for v in raw_ids]
                                if 0 in raw_ids or min(ids) < 0 or max(ids) >= count:
                                    raise ValueError("OBJ face index outside vertices")
                                points = np.array(
                                    [
                                        store.db.execute(
                                            "SELECT x,y,z FROM vertices WHERE level=0 AND key=?", (f"input:{i}",)
                                        ).fetchone()
                                        for i in ids
                                    ]
                                )
                                store.add(0, [f"input:{i}" for i in ids], points, np.array([[0, 1, 2]]))
                            elif words[0] not in {"vt", "vn", "o", "g", "s", "usemtl", "mtllib"}:
                                raise ValueError("unsupported OBJ directive: " + words[0])
                    if _signature(path) != signature:
                        raise ValueError("mesh source changed during import")
                    store.checkpoint_tile(0)
        details = store.finish()
    result = _manifest(metadata, identity, details, 1, sha(output / "mesh.sqlite"))
    atomic_json(output / "MESH.json", result)
    atomic_json(output / "STATUS.json", result)
    return result


class MeshReader:
    """Immutable bundle reader with streaming triangle batches and SHA validation."""

    def __init__(self, path: str | Path):
        self.root = Path(path)
        self.manifest = json.loads((self.root / "MESH.json").read_bytes())
        if self.manifest.get("schema") != "winding-mesh-v1" or self.manifest.get("status") != "MESH_READY":
            raise ValueError("complete mesh bundle required")
        self.file = self.root / "mesh.sqlite"
        self.signature = _signature(self.file)
        self.manifest_sha = sha(self.root / "MESH.json")
        if sha(self.file) != self.manifest["database_sha256"] or _signature(self.file) != self.signature:
            raise ValueError("mesh database hash/identity differs")
        self.db = sqlite3.connect(self.file.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        get = lambda key: json.loads(self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])
        binding = get("binding")
        if (
            not get("complete")
            or binding["metadata"] != self.manifest.get("metadata")
            or binding["identity"] != self.manifest.get("identity")
            or get("summary") != self.manifest.get("levels")
            or json.loads((self.root / "RUN.json").read_bytes()) != binding
        ):
            self.db.close()
            raise ValueError("mesh manifest contradicts committed database binding")
        self.metadata = self.manifest["metadata"]
        aggregates = ",".join(
            f"{op}({alias}.{axis})" for op in ("MIN", "MAX") for alias in ("a", "b", "c") for axis in "xyz"
        )
        row = self.db.execute(
            "SELECT " + aggregates + " FROM faces f JOIN vertices a ON a.id=f.a "
            "JOIN vertices b ON b.id=f.b JOIN vertices c ON c.id=f.c"
        ).fetchone()
        self.bounds = (
            None
            if row[0] is None
            else [np.array(row[:9]).reshape(3, 3).min(0).tolist(), np.array(row[9:]).reshape(3, 3).max(0).tolist()]
        )
        self.validate()

    def validate(self):
        if _signature(self.file) != self.signature or sha(self.root / "MESH.json") != self.manifest_sha:
            raise ValueError("mesh bundle changed during reading")

    def triangles(self, *, batch_size: int = 8192, level: int | None = None):
        batch_size = _integer(batch_size, "batch_size")
        self.validate()
        query = (
            "SELECT a.x,a.y,a.z,b.x,b.y,b.z,c.x,c.y,c.z,f.level FROM faces f "
            "JOIN vertices a ON a.id=f.a JOIN vertices b ON b.id=f.b JOIN vertices c ON c.id=f.c"
        )
        args = () if level is None else (level,)
        if level is not None:
            query += " WHERE f.level=?"
        cursor = self.db.execute(query + " ORDER BY f.id", args)
        while rows := cursor.fetchmany(batch_size):
            self.validate()
            values = np.asarray(rows, float)
            yield values[:, :9].reshape(-1, 3, 3), values[:, 9].astype(np.int64)
        self.validate()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
