"""Pull 3D geometry for the right mushroom body out of MaleCNS v1.0 (neuPrint) for the brain animation.

Output: data/mb_geom.npz with
  <grp>_pts   (N,3) float32  sampled synapse coordinates (dataset voxel units) for grp in kc, dan, mbon, pn
  <grp>_idx   (N,)  int32    index of the parent neuron in the matching *_ids array of data/mb_R.npz
  <grp>_pre   (N,)  bool     True = presynaptic site, False = postsynaptic
  <grp>_comp  (N,)  int8     compartment index into COMP (-1 = outside the 15 lobe compartments, e.g. calyx/peduncle)
  <grp>_soma  (n,3) float32  soma position per neuron (nan when unknown)
No token needed: the /api/custom/custom endpoint is public for this dataset.
"""
import sys, time, numpy as np, requests

URL = "https://neuprint.janelia.org/api/custom/custom"; DS = "male-cns:v1.0"
COMP = ["g1", "g2", "g3", "g4", "g5", "a1", "a2", "a3", "b1", "b2", "a'1", "a'2", "a'3", "b'1", "b'2"]
CIDX = {c: i for i, c in enumerate(COMP)}
# neuPrint compartment strings look like "g5(R)" or "a'2(R)"; map them onto our 15 names
def comp_of(rois):
    # rois = list of ROI flag names on the synapse, e.g. ["CentralBrain", "MB(R)", "gL(R)", "g5(R)"]
    for k in rois or []:
        c = CIDX.get(k.replace("(R)", ""))
        if c is not None: return c
    return -1

def cy(q, tries=4):
    for t in range(tries):
        r = requests.post(URL, json={"cypher": q, "dataset": DS}, timeout=300)
        if r.ok and "data" in r.json(): return r.json()["data"]
        print("retry", t, r.status_code, r.text[:200], file=sys.stderr); time.sleep(3 * (t + 1))
    raise RuntimeError(q[:200])

def synapses(ids, frac, where_extra="", batch=100):
    pts, idx, pre, comp = [], [], [], []
    pos = {int(b): i for i, b in enumerate(ids)}
    for s in range(0, len(ids), batch):
        chunk = list(map(int, ids[s:s + batch]))
        rows = cy(f"MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse) WHERE n.bodyId IN {chunk} AND rand()<{frac} {where_extra} "
                  f"RETURN n.bodyId, s.type, s.location.x, s.location.y, s.location.z, [k IN keys(s) WHERE k ENDS WITH '(R)']")
        for b, ty, x, y, z, c in rows:
            pts.append((x, y, z)); idx.append(pos[int(b)]); pre.append(ty == "pre"); comp.append(comp_of(c))
        print(f"  {s + len(chunk)}/{len(ids)} -> {len(pts)} pts", file=sys.stderr)
    return (np.asarray(pts, np.float32), np.asarray(idx, np.int32), np.asarray(pre, bool), np.asarray(comp, np.int8))

def somas(ids):
    out = np.full((len(ids), 3), np.nan, np.float32); pos = {int(b): i for i, b in enumerate(ids)}
    for s in range(0, len(ids), 300):
        chunk = list(map(int, ids[s:s + 300]))
        for b, loc in cy(f"MATCH (n:Neuron) WHERE n.bodyId IN {chunk} RETURN n.bodyId, n.somaLocation"):
            if loc: out[pos[int(b)]] = loc["coordinates"]
    return out

if __name__ == "__main__":
    d = np.load("data/mb_R.npz", allow_pickle=True); out = {}
    for grp, frac, extra in [("kc", 0.04, ""), ("dan", 0.12, "AND s.type='pre' AND s.`MB(R)`"), ("mbon", 0.10, "AND s.type='post' AND s.`MB(R)`"), ("pn", 0.12, "AND s.type='pre' AND s.`CA(R)`")]:
        print(grp, file=sys.stderr); ids = d[f"{grp}_ids"]
        p, i, r, c = synapses(ids, frac, extra)
        out.update({f"{grp}_pts": p, f"{grp}_idx": i, f"{grp}_pre": r, f"{grp}_comp": c, f"{grp}_soma": somas(ids)})
    np.savez_compressed("data/mb_geom.npz", **out)
    print({k: v.shape for k, v in out.items()})
