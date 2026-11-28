"""Cut every handle of a sheet piece along its own short loop; holes stay holes.

A level surface at the field's pitch has tunnels a few voxels wide (PHerc0175A fine
field: about 3.5 handles and 89 holes per cm2). A piece with handles is not a planar
domain, and ``igl.cut_to_disk`` joins all of them through long seams that the map then
has to open: the whole-sheet map of such a piece collapses. Cutting each handle around
its tunnel instead turns it into two small holes and leaves the rest of the sheet whole.

Method, on the surface with every boundary loop contracted to one node, so a loop may
pass through a hole for free but never just run around one:

1. Sites. One shortest-path tree from a root and the cotree that keeps the longest
   loops leave two edges per handle: the greedy loops through the root (Erickson and
   Whittlesey 2005). Each of those loops reaches its handle, and the edge that closes it
   lies on the far side of the handle.
2. Local loops. Around each site, a shortest-path tree limited to a radius. The loop a
   non-tree edge closes separates the surface exactly when its dual edge is a bridge of
   the dual graph without the tree; everything beyond the radius counts as one face.
   The shortest non-separating loop is cut. Its vertices and the holes it passes are
   then closed to later loops, so every loop is a simple closed curve away from all
   earlier cuts and lowers the genus by one.
3. A ball cannot see a cut beyond its radius, so two loops may cut off the part between
   them: loops that only join two parts are given up until one part is left. The cuts
   are applied and checked (one piece, genus lower by the number of loops), and the
   search repeats, with a doubled radius when a round found nothing.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import (
    breadth_first_order,
    connected_components,
    dijkstra,
    minimum_spanning_tree,
)

# First search radius, in median edge lengths, and how often it may double. A loop of
# length L is found from a radius of L/2, so short loops are cut before any long one.
RADIUS_EDGES = 4
DOUBLINGS = 6
# How often a search may move onto the loop it found to look for a shorter one.
RECENTRE = 4
# Faces within this many loop lengths of a handle loop leave the piece with it.
COLLAR = 0.25


def _on_cycle(n, a, b, rng):
    """For undirected edges ``(a[i], b[i])`` on n nodes: True where the edge lies on a cycle.

    An edge outside a spanning forest closes one cycle; a random mark on its two ends,
    summed (xor) over each subtree, is non-zero exactly on the forest edges of that cycle.
    """
    out = np.zeros(len(a), bool)
    ids = np.flatnonzero(a != b)
    if not len(ids):
        return out
    lo, hi = np.minimum(a[ids], b[ids]), np.maximum(a[ids], b[ids])
    pairs, inverse, counts = np.unique(lo * n + hi, return_inverse=True, return_counts=True)
    lo, hi = pairs // n, pairs % n
    count, label = connected_components(
        csr_matrix((np.ones(len(pairs)), (lo, hi)), shape=(n, n)), directed=False
    )
    roots = np.empty(count, np.int64)
    roots[label[::-1]] = np.arange(n)[::-1]
    # One extra node joins the components, so one search covers the forest.
    graph = csr_matrix(
        (np.ones(len(pairs) + count), (np.r_[lo, np.full(count, n)], np.r_[hi, roots])),
        shape=(n + 1, n + 1),
    )
    order, pred = breadth_first_order(graph + graph.T, n, directed=False)
    in_forest = (pred[lo] == hi) | (pred[hi] == lo)
    value = np.zeros(n + 1, np.uint64)
    extra = np.flatnonzero(~in_forest)
    marks = rng.integers(1, np.iinfo(np.int64).max, len(extra)).astype(np.uint64)
    np.bitwise_xor.at(value, lo[extra], marks)
    np.bitwise_xor.at(value, hi[extra], marks)
    position = np.empty(n + 1, np.int64)
    position[order] = np.arange(len(order))
    # Parents come in non-decreasing position along a breadth-first order, so the nodes
    # after the last one's parent can be folded into their parents in one step.
    parent_position = position[pred[order[1:]]]
    stop = len(order)
    while stop > 1:
        start = max(int(parent_position[stop - 2]) + 1, 1)
        nodes = order[start:stop]
        np.bitwise_xor.at(value, pred[nodes], value[nodes])
        stop = start
    child = np.where(pred[lo] == hi, lo, hi)
    out[ids] = (np.where(in_forest, value[child] != 0, True) | (counts > 1))[inverse]
    return out


def _row_starts(graph, keep):
    """Row pointers of a CSR matrix after dropping the entries where ``keep`` is False."""
    return np.r_[0, np.cumsum(keep)][graph.indptr]


class _Surface:
    """Edges, dual faces and the contracted graph of one manifold piece."""

    def __init__(self, vertices, faces, max_rim):
        import igl

        f = faces.astype(np.int64, copy=False)
        nv = len(vertices)
        self.f, self.nv = f, nv
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        owner = np.tile(np.arange(len(f)), 3)
        key = np.minimum(e[:, 0], e[:, 1]) * nv + np.maximum(e[:, 0], e[:, 1])
        ukey, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        if counts.max() > 2:
            raise ValueError("piece is not edge-manifold")
        self.edges = np.column_stack([ukey // nv, ukey % nv])
        self.face_edges = inverse.reshape(3, len(f)).T
        self.interior = counts == 2
        self.cut = np.zeros(len(ukey), bool)
        order = np.argsort(inverse, kind="stable")
        first = np.r_[0, np.cumsum(counts)[:-1]]
        self.face_a = owner[order][first]
        self.face_b = np.where(
            self.interior, owner[order][np.minimum(first + 1, len(order) - 1)], -1
        )
        self.length = np.linalg.norm(
            vertices[self.edges[:, 0]] - vertices[self.edges[:, 1]], axis=1
        )
        loops = [np.asarray(loop, np.int64) for loop in igl.boundary_loop_all(f)]
        node = np.arange(nv, dtype=np.int64)
        for k, loop in enumerate(loops):
            node[loop] = nv + k
        self.node, self.total = node, nv + len(loops)
        self.na, self.nb = node[self.edges[:, 0]], node[self.edges[:, 1]]
        # The shortest edge between two nodes carries that link of the graph.
        cand = np.flatnonzero(self.interior & (self.na != self.nb))
        lo = np.minimum(self.na[cand], self.nb[cand])
        hi = np.maximum(self.na[cand], self.nb[cand])
        best = np.lexsort((self.length[cand], hi, lo))
        cand, lo, hi = cand[best], lo[best], hi[best]
        firsts = np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])]
        self.rep, self.rep_key = cand[firsts], (lo * self.total + hi)[firsts]
        graph = coo_matrix(
            (self.length[self.rep] + 1e-9, (lo[firsts], hi[firsts])), shape=(self.total,) * 2
        ).tocsr()
        self.graph = (graph + graph.T).tocsr()
        self.graph.sort_indices()
        ids = np.flatnonzero(self.interior)
        ends = np.r_[self.na[ids], self.nb[ids]]
        by = np.argsort(ends, kind="stable")
        self.at_edge = np.r_[ids, ids][by]
        self.at_ptr = np.searchsorted(ends[by], np.arange(self.total + 1))
        self.blocked = np.zeros(self.total, bool)
        # Long rims (the outline, the ends of a wrap) carry no local loop. Each closed rim
        # adds one site with nothing to find, so only those rims are closed.
        for k, loop in enumerate(loops):
            rim = np.linalg.norm(vertices[np.roll(loop, -1)] - vertices[loop], axis=1).sum()
            if rim > max_rim:
                self.block(nv + k)
        self.genus = (2 - (nv - len(self.edges) + len(f)) - len(loops)) // 2

    def block(self, x):
        """Close a node to every later path."""
        if self.blocked[x]:
            return
        self.blocked[x] = True
        g = self.graph
        row = slice(g.indptr[x], g.indptr[x + 1])
        for y in g.indices[row]:
            lo = g.indptr[y]
            g.data[lo + np.searchsorted(g.indices[lo : g.indptr[y + 1]], x)] = np.inf
        g.data[row] = np.inf

    def links(self, x, y):
        key = np.minimum(x, y) * self.total + np.maximum(x, y)
        return self.rep[np.searchsorted(self.rep_key, key)]

    def sites(self):
        """Edges that close the greedy loops through a root, shortest loop first.

        A closed rim can split the graph (and each vertex of a rim is a node with no link),
        so every part with a link gets its own root and tree.
        """
        graph = self.graph
        open_links = np.isfinite(graph.data)
        linked = csr_matrix(
            (
                np.ones(int(open_links.sum())),
                graph.indices[open_links],
                _row_starts(graph, open_links),
            ),
            shape=graph.shape,
        )
        _, part = connected_components(linked, directed=False)
        first = np.unique(part, return_index=True)[1]
        roots = first[np.bincount(part)[part[first]] > 1]
        if not len(roots):
            return np.zeros(0, np.int64)
        dist, pred, _ = dijkstra(
            graph, directed=False, indices=roots, return_predecessors=True, min_only=True
        )
        reached = np.flatnonzero((pred >= 0) & np.isfinite(dist))
        in_tree = np.zeros(len(self.edges), bool)
        in_tree[self.links(reached, pred[reached])] = True
        free = np.flatnonzero(self.interior & ~in_tree)
        loop = dist[self.na[free]] + dist[self.nb[free]] + self.length[free]
        finite = np.isfinite(loop)
        if not finite.any():
            return free[:0]
        # The longest loops enter the cotree first, so the shortest stay out of it.
        weight = np.where(finite, loop[finite].max() - loop, -1.0) + 2.0
        nf = len(self.f)
        a, b = self.face_a[free], self.face_b[free]
        tree = minimum_spanning_tree(coo_matrix((weight, (a, b)), shape=(nf, nf)).tocsr()).tocoo()
        row, col = tree.row.astype(np.int64), tree.col.astype(np.int64)
        used = np.minimum(row, col) * nf + np.maximum(row, col)
        left = ~np.isin(np.minimum(a, b) * nf + np.maximum(a, b), used) & finite
        return free[left][np.argsort(loop[left], kind="stable")]

    def local_loop(self, source, radius, rng):
        """A short non-separating loop near ``source``: (length, edge ids, nodes) or None.

        The shortest such loop through a node is always closed by one edge on that node's
        shortest-path tree, but a loop away from the node is not. So the search moves
        onto the loop it finds and repeats while that gives a shorter one.
        """
        best = self._tree_loop(source, radius, rng)
        for _ in range(RECENTRE):
            if best is None or source in best[2]:
                break
            source = int(best[2][0])
            again = self._tree_loop(source, radius, rng)
            if again is None or again[0] >= best[0]:
                break
            best = again
        return best

    def _tree_loop(self, source, radius, rng):
        """Shortest non-separating loop that one edge closes on the tree of ``source``."""
        dist, pred = dijkstra(
            self.graph, directed=False, indices=source, return_predecessors=True, limit=radius
        )
        inside = np.isfinite(dist)
        nodes = np.flatnonzero(inside)
        if len(nodes) < 3:
            return None
        near = np.unique(
            np.concatenate([self.at_edge[self.at_ptr[x] : self.at_ptr[x + 1]] for x in nodes])
        )
        faces = np.unique(np.r_[self.face_a[near], self.face_b[near]])
        # A face with a node beyond the radius is outside; one at an earlier cut or at a
        # closed rim stays its own face, so a loop around that cut still separates.
        far = ~inside & ~self.blocked
        faces = faces[~far[self.node[self.f[faces]]].any(axis=1)]
        if not len(faces):
            return None
        ids = np.unique(self.face_edges[faces])
        others = nodes[nodes != source]
        tree = self.links(others, pred[others])
        free = ids[self.interior[ids] & ~self.cut[ids] & ~np.isin(ids, tree)]
        if not len(free):
            return None

        def local(face):  # the last node stands for every face outside the ball
            at = np.minimum(np.searchsorted(faces, face), len(faces) - 1)
            return np.where(faces[at] == face, at, len(faces))

        cyclic = _on_cycle(len(faces) + 1, local(self.face_a[free]), local(self.face_b[free]), rng)
        cand = free[cyclic & inside[self.na[free]] & inside[self.nb[free]]]
        if not len(cand):
            return None
        # Each candidate closes the two tree paths from its ends up to where they meet.
        # Climb the end that is farther from the source until both are the same node.
        a, b = self.na[cand].copy(), self.nb[cand].copy()
        while True:
            apart = a != b
            if not apart.any():
                break
            up = apart & (dist[a] >= dist[b])
            a[up] = pred[a[up]]
            down = apart & ~up
            b[down] = pred[b[down]]
        size = dist[self.na[cand]] + dist[self.nb[cand]] + self.length[cand] - 2 * dist[a]
        best = int(np.argmin(size))
        edge, meet = int(cand[best]), int(a[best])
        paths = []
        for x in (int(self.na[edge]), int(self.nb[edge])):
            path = [x]
            while x != meet:
                x = int(pred[x])
                path.append(x)
            paths.append(np.asarray(path, np.int64))
        loop = np.r_[
            edge, self.links(paths[0][:-1], paths[0][1:]), self.links(paths[1][:-1], paths[1][1:])
        ]
        return float(self.length[loop].sum()), loop, np.union1d(paths[0], paths[1])


def cut_handles(vertices, faces, check=None, *, radius_edges=RADIUS_EDGES, seed=0):
    """Cut the handles of one manifold piece along short loops.

    Returns ``(vertices, faces, parent, report)``: the cut piece, the input vertex of
    each of its vertices, and what was cut (``loop_vertices``: the input vertices of each
    loop). Faces keep their order. Handles with no loop inside the last radius are left
    to the caller (``report["genus_left"]``).
    """
    import igl

    from ..tifxyz_guide import _cut_flags

    rng = np.random.default_rng(seed)
    v, f = vertices, faces.astype(np.int64, copy=False)
    parent = np.arange(len(v))
    edge = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
    radius = radius_edges * float(np.median(edge))
    last_radius = radius * 2**DOUBLINGS
    report = dict(genus=0, loops=0, loop_length=[], loop_vertices=[], seam_length=0.0, rounds=[])
    genus = None
    while genus is None or genus > 0:
        surface = _Surface(v, f, 4 * last_radius)
        if genus is None:
            report["genus"] = int(surface.genus)
        genus = int(surface.genus)
        if genus <= 0:
            break
        found = []
        for site in surface.sites().tolist():
            if len(found) == genus:
                break
            if check:
                check()
            for source in (int(surface.na[site]), int(surface.nb[site])):
                while len(found) < genus and not surface.blocked[source]:
                    loop = surface.local_loop(source, radius, rng)
                    if loop is None:
                        break
                    found.append((loop[0], loop[1], parent[loop[2][loop[2] < len(v)]]))
                    surface.cut[loop[1]] = True
                    for x in loop[2].tolist():
                        surface.block(x)
        entry = dict(radius=radius, genus=genus, found=len(found))
        report["rounds"].append(entry)
        # Give up the loops that only join two parts, longest first, until one is left.
        open_edges = np.flatnonzero(surface.interior & ~surface.cut)
        parts, label = connected_components(
            coo_matrix(
                (
                    np.ones(len(open_edges)),
                    (surface.face_a[open_edges], surface.face_b[open_edges]),
                ),
                shape=(len(f),) * 2,
            ),
            directed=False,
        )
        group = np.arange(parts)

        def find(x):
            while group[x] != x:
                group[x] = group[group[x]]
                x = group[x]
            return int(x)

        kept = []
        for size, loop, where in sorted(found, key=lambda item: -item[0]):
            sides = {
                find(x) for x in np.r_[label[surface.face_a[loop]], label[surface.face_b[loop]]]
            }
            if len(sides) > 1:
                first = sides.pop()
                for other in sides:
                    group[other] = first
            else:
                kept.append((size, loop, where))
        entry["loops"] = len(kept)
        if not kept:
            if radius >= last_radius:
                break
            radius *= 2
            continue
        cut = np.unique(np.concatenate([loop for _, loop, _ in kept]))
        cv, cf, cut_parent = igl.cut_mesh(v, f, _cut_flags(f, list(surface.edges[cut]), len(v)))
        cf = np.asarray(cf, np.int64)
        pieces, _ = connected_components(
            coo_matrix(
                (np.ones(cf.size), (cf.ravel(), np.roll(cf, -1, axis=1).ravel())),
                shape=(len(cv),) * 2,
            ),
            directed=False,
        )
        rims = len(igl.boundary_loop_all(cf))
        after = (2 - int(igl.euler_characteristic(cf)) - rims) // 2
        if pieces != 1 or after != genus - len(kept):
            # Not what the local test promised: keep the piece as it was before this round.
            entry["rejected"] = dict(pieces=int(pieces), genus=int(after))
            break
        v, f, parent, genus = np.asarray(cv), cf, parent[np.asarray(cut_parent)], after
        report["loops"] += len(kept)
        report["loop_length"] += [size for size, _, _ in kept]
        report["loop_vertices"] += [where for _, _, where in kept]
        report["seam_length"] += float(surface.length[cut].sum())
    report["genus_left"] = int(report["genus"] - report["loops"])
    return v, f, parent, report


def handle_collars(vertices, faces, check=None, *, collar=COLLAR, radius_edges=RADIUS_EDGES):
    """Faces to set aside so that a piece has no handle left: a collar around each loop.

    A tunnel cannot lie flat with the sheet it pierces, cut or not: its wall and the two
    funnels around its mouths would be stretched however the map is made. The faces
    within ``collar`` loop lengths of each handle loop (along the surface) are therefore
    taken out of the piece, which removes the tunnel and leaves an ordinary hole at each
    mouth; the caller flattens them as small pieces of their own.

    Returns ``(aside, report)``: one flag per face and the ``cut_handles`` report.
    """
    f = faces.astype(np.int64, copy=False)
    _, _, _, report = cut_handles(vertices, f, check, radius_edges=radius_edges)
    aside = np.zeros(len(f), bool)
    loops = report.pop("loop_vertices")
    if not loops:
        return aside, report
    reach = collar * np.asarray(report["loop_length"])
    n = len(vertices)
    tail, head = f.ravel(), np.roll(f, -1, axis=1).ravel()
    key = np.unique(np.minimum(tail, head) * n + np.maximum(tail, head))
    e = np.column_stack([key // n, key % n])
    length = np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1) + 1e-9
    # One extra node starts every loop at once, later for the loops with a shorter reach.
    start = np.concatenate([np.full(len(loop), n) for loop in loops])
    delay = np.concatenate(
        [np.full(len(loop), reach.max() - r + 1e-9) for loop, r in zip(loops, reach)]
    )
    graph = coo_matrix(
        (np.r_[length, delay], (np.r_[e[:, 0], start], np.r_[e[:, 1], np.concatenate(loops)])),
        shape=(n + 1, n + 1),
    ).tocsr()
    dist = dijkstra(graph, directed=False, indices=n, limit=reach.max() + 2e-9)
    on_loop = np.zeros(n, bool)
    on_loop[np.concatenate(loops)] = True
    # The faces at the loop itself always go, or a loop of three edges would stay closed.
    aside = np.isfinite(dist[:n])[f].all(axis=1) | on_loop[f].any(axis=1)
    report["collar_faces"] = int(aside.sum())
    return aside, report
