"""Pull whole-brain outline geometry from MaleCNS v1.0 (neuPrint) for the brain animation.

Output: data/brain_mesh.npz with
  brain_pts (N,3) float32  subsampled surface vertices of the CentralBrain ROI mesh
  ol_pts    (M,3) float32  subsampled surface vertices of the optic-lobe neuropils (LA, ME, LO, LOP, AME), both sides
  kcL_pts   (K,3) float32  sampled synapse positions of LEFT-hemisphere Kenyon cells (dim mirror of the studied right MB)
Same voxel coordinate frame as data/mb_geom.npz. No token needed.
"""
import sys, numpy as np, requests

MESH = "https://neuprint.janelia.org/api/roimeshes/mesh/male-cns:v1.0/{}"
URL = "https://neuprint.janelia.org/api/custom/custom"; DS = "male-cns:v1.0"

def mesh_vertices(roi, keep, seed=0):
    r = requests.get(MESH.format(roi), timeout=300); r.raise_for_status()
    V = np.array([list(map(float, l.split()[1:4])) for l in r.text.splitlines() if l.startswith("v ")], np.float32)
    rng = np.random.default_rng(seed); idx = rng.choice(len(V), size=min(keep, len(V)), replace=False)
    print(f"  {roi}: {len(V)} verts -> {len(idx)}", file=sys.stderr); return V[idx]

def cy(q):
    r = requests.post(URL, json={"cypher": q, "dataset": DS}, timeout=300); r.raise_for_status(); return r.json()["data"]

if __name__ == "__main__":
    brain = mesh_vertices("CentralBrain", 45000)
    ol = np.concatenate([mesh_vertices(f"{n}({s})", 9000) for s in "RL" for n in ["LA", "ME", "LO", "LOP", "AME"]])
    ids = [b for (b,) in cy("MATCH (n:Neuron) WHERE n.type STARTS WITH 'KC' AND n.somaSide='L' RETURN n.bodyId")]
    print(f"  left KCs: {len(ids)}", file=sys.stderr); pts = []
    for s in range(0, len(ids), 150):
        chunk = ids[s:s + 150]
        pts += [(x, y, z) for x, y, z in cy(f"MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse) WHERE n.bodyId IN {chunk} AND rand()<0.02 "
                                          "RETURN s.location.x, s.location.y, s.location.z")]
    kcL = np.asarray(pts, np.float32); print(f"  left KC synapses sampled: {len(kcL)}", file=sys.stderr)
    np.savez_compressed("data/brain_mesh.npz", brain_pts=brain, ol_pts=ol, kcL_pts=kcL)
    print({"brain": brain.shape, "ol": ol.shape, "kcL": kcL.shape})
