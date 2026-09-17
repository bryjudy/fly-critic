"""Pull the right-hemisphere mushroom body circuit out of MaleCNS v1.0 (neuPrint).

Output: data/mb_R.npz  with
  pn_ids, kc_ids, mbon_ids, dan_ids            neuron bodyIds
  pn_types, kc_types, mbon_types, dan_types    cell-type strings
  W_pn_kc   (n_pn, n_kc)     synapse counts PN -> KC
  W_kc_mbon (n_kc, n_mbon, n_comp)  synapse counts KC -> MBON, split by compartment
  W_apl_kc  (n_kc,)          APL -> KC synapse counts (feedback inhibition)
  W_kc_apl  (n_kc,)          KC -> APL
  W_mbon_dan (n_mbon, n_dan) MBON -> DAN feedback
  W_dan_mbon (n_dan, n_mbon) DAN -> MBON direct
  dan_comp  (n_dan, n_comp)  DAN presynaptic output fraction per compartment
  mbon_comp (n_mbon, n_comp) MBON postsynaptic input fraction per compartment
  dan_family (n_dan,)        0 = PAM (reward), 1 = PPL1 (punishment)
  compartments               list of 15 compartment names
No token needed: the /api/custom/custom endpoint is public for this dataset.
"""
import json, sys, time
import numpy as np, requests

URL = "https://neuprint.janelia.org/api/custom/custom"
DS = "male-cns:v1.0"
SIDE = "R"
COMP = ["g1", "g2", "g3", "g4", "g5", "a1", "a2", "a3", "b1", "b2", "a'1", "a'2", "a'3", "b'1", "b'2"]
CIDX = {c: i for i, c in enumerate(COMP)}

def cy(q, tries=4):
    for t in range(tries):
        r = requests.post(URL, json={"cypher": q, "dataset": DS}, timeout=300)
        if r.ok:
            d = r.json()
            if "data" in d:
                return d["data"]
        print("retry", t, r.status_code, r.text[:200], file=sys.stderr); time.sleep(3 * (t + 1))
    raise RuntimeError(q[:200])

def fam(t):
    if t.startswith("KC"): return "KC"
    if t.startswith("MBON"): return "MBON"
    if t.startswith("PAM"): return "PAM"
    if t.startswith("PPL1"): return "PPL1"
    if t in ("APL", "DPM"): return t
    return "PN"

print("neurons...")
rows = cy(f"""
MATCH (n:Neuron) WHERE n.somaSide='{SIDE}' AND (
  n.type =~ '(KC|MBON|PAM|PPL1).*' OR n.type IN ['APL','DPM'] OR n.type =~ '.*_[a-z0-9]*PN[0-9]*$')
RETURN n.bodyId, n.type, n.instance, n.predictedNt, n.roiInfo""")
neurons = {}
for bid, t, inst, nt, roi in rows:
    neurons[bid] = dict(type=t, inst=inst, nt=nt, fam=fam(t), roi=json.loads(roi or "{}"))
byfam = {}
for b, n in neurons.items(): byfam.setdefault(n["fam"], []).append(b)
print({k: len(v) for k, v in byfam.items()})

# keep only PNs that actually feed KCs (olfactory + thermo/hygro PNs) — determined by edges below
ids = list(neurons)
print("edges...")
edges = []
CH = 400
for i in range(0, len(ids), CH):
    chunk = ids[i:i + CH]
    rows = cy(f"""
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
WHERE a.bodyId IN {chunk} AND b.somaSide='{SIDE}' AND (
  b.type =~ '(KC|MBON|PAM|PPL1).*' OR b.type IN ['APL','DPM'])
RETURN a.bodyId, b.bodyId, c.weight, c.roiInfo""")
    edges += rows
    print(f"  {i+len(chunk)}/{len(ids)} -> {len(edges)} edges")

def comp_split(roi):
    """per-compartment postsynaptic count from an edge roiInfo dict (right side)."""
    out = np.zeros(len(COMP))
    for k, v in roi.items():
        name = k.split("(")[0]
        if name in CIDX and k.endswith(f"({SIDE})"):
            out[CIDX[name]] += v.get("post", 0)
    return out

# PN set = candidate PNs with >= 3 synapses onto any KC
pn_set = set()
for a, b, w, roi in edges:
    if neurons[a]["fam"] == "PN" and neurons[b]["fam"] == "KC" and w >= 3: pn_set.add(a)
order = lambda f: sorted(byfam[f], key=lambda b: (neurons[b]["type"], b))
pn_ids = sorted(pn_set, key=lambda b: (neurons[b]["type"], b))
kc_ids, mbon_ids = order("KC"), order("MBON")
dan_ids = order("PAM") + order("PPL1")
apl = byfam["APL"][0]
ix = {f: {b: i for i, b in enumerate(l)} for f, l in
      dict(PN=pn_ids, KC=kc_ids, MBON=mbon_ids, DAN=dan_ids).items()}
nP, nK, nM, nD, nC = len(pn_ids), len(kc_ids), len(mbon_ids), len(dan_ids), len(COMP)
print(dict(PN=nP, KC=nK, MBON=nM, DAN=nD))

W_pn_kc = np.zeros((nP, nK)); W_kc_mbon = np.zeros((nK, nM, nC)); W_kc_kc = np.zeros((nK, nK))
W_apl_kc = np.zeros(nK); W_kc_apl = np.zeros(nK)
W_mbon_dan = np.zeros((nM, nD)); W_dan_mbon = np.zeros((nD, nM)); W_dan_kc = np.zeros((nD, nK))
W_mbon_mbon = np.zeros((nM, nM))
unassigned = 0
for a, b, w, roi in edges:
    fa, fb = neurons[a]["fam"], neurons[b]["fam"]
    roi = json.loads(roi or "{}")
    if fa == "PN" and fb == "KC" and a in pn_set: W_pn_kc[ix["PN"][a], ix["KC"][b]] += w
    elif fa == "KC" and fb == "MBON":
        cs = comp_split(roi)
        if cs.sum() == 0: unassigned += w; continue
        W_kc_mbon[ix["KC"][a], ix["MBON"][b]] += w * cs / cs.sum()
    elif fa == "KC" and fb == "KC": W_kc_kc[ix["KC"][a], ix["KC"][b]] += w
    elif a == apl and fb == "KC": W_apl_kc[ix["KC"][b]] += w
    elif fa == "KC" and b == apl: W_kc_apl[ix["KC"][a]] += w
    elif fa == "MBON" and fb in ("PAM", "PPL1"): W_mbon_dan[ix["MBON"][a], ix["DAN"][b]] += w
    elif fa in ("PAM", "PPL1") and fb == "MBON": W_dan_mbon[ix["DAN"][a], ix["MBON"][b]] += w
    elif fa in ("PAM", "PPL1") and fb == "KC": W_dan_kc[ix["DAN"][a], ix["KC"][b]] += w
    elif fa == "MBON" and fb == "MBON": W_mbon_mbon[ix["MBON"][a], ix["MBON"][b]] += w
print("KC->MBON synapses outside named compartments:", unassigned, "of", W_kc_mbon.sum() + unassigned)

def comp_frac(bid, key):
    out = np.zeros(nC)
    for k, v in neurons[bid]["roi"].items():
        name = k.split("(")[0]
        if name in CIDX and k.endswith(f"({SIDE})"): out[CIDX[name]] += v.get(key, 0)
    return out
dan_comp_abs = np.stack([comp_frac(b, "pre") for b in dan_ids])
mbon_comp_abs = np.stack([comp_frac(b, "post") for b in mbon_ids])
dan_comp = dan_comp_abs / np.maximum(dan_comp_abs.sum(1, keepdims=True), 1)
mbon_comp = mbon_comp_abs / np.maximum(mbon_comp_abs.sum(1, keepdims=True), 1)
dan_family = np.array([0 if neurons[b]["fam"] == "PAM" else 1 for b in dan_ids])

np.savez_compressed("data/mb_R.npz",
    pn_ids=pn_ids, kc_ids=kc_ids, mbon_ids=mbon_ids, dan_ids=dan_ids,
    pn_types=[neurons[b]["type"] for b in pn_ids], kc_types=[neurons[b]["type"] for b in kc_ids],
    mbon_types=[neurons[b]["type"] for b in mbon_ids], dan_types=[neurons[b]["type"] for b in dan_ids],
    mbon_inst=[neurons[b]["inst"] for b in mbon_ids], dan_inst=[neurons[b]["inst"] for b in dan_ids],
    mbon_nt=[neurons[b]["nt"] or "unclear" for b in mbon_ids],
    W_pn_kc=W_pn_kc, W_kc_mbon=W_kc_mbon, W_kc_kc=W_kc_kc, W_apl_kc=W_apl_kc, W_kc_apl=W_kc_apl,
    W_mbon_dan=W_mbon_dan, W_dan_mbon=W_dan_mbon, W_dan_kc=W_dan_kc, W_mbon_mbon=W_mbon_mbon,
    dan_comp=dan_comp, mbon_comp=mbon_comp, dan_comp_abs=dan_comp_abs, mbon_comp_abs=mbon_comp_abs, dan_family=dan_family, compartments=COMP)
print("saved data/mb_R.npz")
print("PN->KC total syn", W_pn_kc.sum(), " mean PN inputs/KC", (W_pn_kc > 0).sum(0).mean())
print("KC->MBON per compartment:", dict(zip(COMP, W_kc_mbon.sum((0, 1)).round())))
print("DAN family per compartment (PAM frac):",
      dict(zip(COMP, ((dan_comp_abs * (dan_family == 0)[:, None]).sum(0) / np.maximum(dan_comp_abs.sum(0), 1e-9)).round(2))))
