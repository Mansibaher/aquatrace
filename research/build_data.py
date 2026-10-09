"""Reproduce AquaTrace: EPANET steady states, independent scenario split, RF export."""
from pathlib import Path
import json
import tempfile
import hashlib
import numpy as np
import wntr
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
SENSORS = [f'Z{z}_S{s}' for z in (1, 2) for s in (1, 2, 3)]
SOURCE = ROOT / 'research/network.inp'
rng = np.random.default_rng(42)
cache = {}

def simulate(node, coeff, demand):
    key = (node, coeff, demand)
    if key not in cache:
        network = wntr.network.WaterNetworkModel(str(SOURCE))
        network.options.hydraulic.demand_multiplier = demand
        if node != 'none':
            network.get_node(node).emitter_coefficient = coeff
        with tempfile.TemporaryDirectory() as tmp:
            result = wntr.sim.EpanetSimulator(network).run_sim(
                file_prefix=str(Path(tmp) / 'run'), convergence_error=True)
        p = result.node['pressure'].iloc[0][SENSORS].to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError(f'Invalid pressures: {key}: {p}')
        cache[key] = p
    return cache[key]

def dataset(demands, coeffs, samples):
    X, y = [], []
    # Equal leak/no-leak counts for transparent balanced evaluation.
    for demand in demands:
        base = simulate('none', 0, demand)
        for node in SENSORS:
            for coeff in coeffs:
                residual = base - simulate(node, coeff, demand)
                for _ in range(samples):
                    sigma = rng.uniform(0, .5)
                    X.extend([residual + rng.normal(0, sigma, 6), rng.normal(0, sigma, 6)])
                    y.extend([1, 0])
    return np.asarray(X), np.asarray(y)

print('Simulating training operating conditions...', flush=True)
X, y = dataset([.6, .85, 1.15, 1.4], [.00007, .00015, .00035, .0007, .0012], 12)
model = RandomForestClassifier(n_estimators=48, max_depth=7, min_samples_leaf=5,
                               random_state=42, n_jobs=-1).fit(X, y)
print('Simulating held-out operating conditions...', flush=True)
Xt, yt = dataset([.7, 1., 1.3], [.0001, .0003, .001], 16)
pred = model.predict(Xt)
cm = confusion_matrix(yt, pred, labels=[0, 1]).tolist()
trees=[]
for estimator in model.estimators_:
    t=estimator.tree_
    values=t.value[:, 0, :]
    probs=values[:, 1] / values.sum(axis=1)
    trees.append(dict(left=t.children_left.tolist(), right=t.children_right.tolist(),
                      feature=t.feature.tolist(), threshold=t.threshold.tolist(), value=probs.tolist()))
scenarios=[]
for demand in [.7, 1., 1.3]:
    base=simulate('none',0,demand)
    scenarios.append(dict(node='none',size=0,demand=demand,baseline=base.tolist(),pressure=base.tolist(),flow=0))
    for node in SENSORS:
        for size, coeff in enumerate([.0001, .0003, .001]):
            p=simulate(node,coeff,demand)
            scenarios.append(dict(node=node,size=size,demand=demand,baseline=base.tolist(),pressure=p.tolist(),
                                  flow=float(coeff*np.sqrt(p[SENSORS.index(node)])*1000)))
report=dict(training_rows=len(y),test_rows=len(yt),train_demands=[.6,.85,1.15,1.4],
            test_demands=[.7,1,1.3],train_coefficients=[.00007,.00015,.00035,.0007,.0012],
            test_coefficients=[.0001,.0003,.001],confusion_matrix=cm,
            precision=float(precision_score(yt,pred)),recall=float(recall_score(yt,pred)),
            f1=float(f1_score(yt,pred)),feature_importance=model.feature_importances_.tolist(),
            seed=42,wntr=wntr.__version__,sklearn=sklearn.__version__,
            network_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            limitation='Same-network synthetic evaluation, held-out demand and emitter coefficient combinations. Not real-world or unseen-network validation.')
data=dict(version=1,sensors=SENSORS,scenarios=scenarios,trees=trees,evaluation=report)
(ROOT/'dist/data.json').write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
(ROOT/'dist/evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
# Independent exported-model parity fixtures for the JavaScript inference test.
fixture=dict(features=Xt[:100].tolist(),scores=model.predict_proba(Xt[:100])[:,1].tolist())
(ROOT/'research/parity.json').write_text(json.dumps(fixture),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
