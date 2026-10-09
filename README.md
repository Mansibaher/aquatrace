# AquaTrace — Water Leak Lab

An interactive research prototype built from Mansi Aher's original two-zone EPANET network. Choose a leak location, emitter size, demand, and sensor noise; inspect pressure residuals and Random Forest predictions; replay and save experiments locally.

## Run the interactive lab

No installation or API key is needed to use the bundled browser app. From the repository root:

```sh
python -m http.server 8766 --directory dist
```

Open http://localhost:8766. Choose a leak scenario, adjust demand and sensor noise, then inspect predictions and save experiments in the browser notebook.

## Architecture

Original EPANET INP → Python WNTR/EPANET steady-state simulation → independent operating-condition train/test split → scikit-learn Random Forest → exported trees and hydraulic scenario bank → browser inference, SVG visualizations and device-local notebook.

The shipped app is static and needs no API key. It contains 57 exact precomputed hydraulic states, not a runtime EPANET solver. Noise and model inference execute in the browser. Notebook data stays in localStorage.

## Reproduce

Use Python 3.12 and Node.js. Create a virtual environment and install `research/requirements.txt`. Run `python research/build_data.py` from this directory, then `npm test`. Serve `dist` with `python -m http.server 8766 --directory dist` and open http://localhost:8766.

## Evidence and limitations

`dist/evaluation.json` records the actual evaluation, versions, seed and source-network SHA-256. `research/parity.json` contains sklearn prediction fixtures used to verify JavaScript inference. The evaluation holds out demand and emitter coefficient combinations; all observations still use the same physical network topology. Training and test sets are balanced and include Gaussian noise.

This newly trained demo model is **not the original thesis model**. No physical validation or thesis accuracy claim is made. The replay switches between two steady states, not a transient hydraulic process. Steps are observations, not seconds. Demand is known, enabling a matched no-leak baseline. No leak localization model is claimed: the highlighted leak is user-injected ground truth.

Future work: unknown-demand robustness, sensor drift/dropout, extended-period simulation, held-out networks, physical validation, and an on-demand simulation backend.

## Attribution

Network source: `TwoZone_3Sensors_SignificantDistance.inp`, from Mansi Aher's thesis workspace; copied unchanged as `research/network.inp`. Simulation engine: EPA WNTR / EPANET 2.2. See https://usepa.github.io/WNTR/.
