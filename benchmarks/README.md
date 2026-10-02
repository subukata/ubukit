# Benchmarks

- `python/smoke/`: small parameter-search and quality checks
- `python/efficiency/`: SOM comparisons and replay of recorded failure conditions
- `javascript/`: comparison drivers for the JavaScript implementation

Build and install the local packages before running a driver; see [getting started](../docs/getting-started.md). JavaScript drivers import the installed package from `javascript/consumer/node_modules/ubukit-js`. Some comparisons also use fixed reference implementations under `javascript/validation/`.

After building the JavaScript tarball, install it in the benchmark consumer directory from the repository root:

```sh
cd javascript/consumer
npm install --offline --ignore-scripts --no-audit --no-fund --package-lock=false
cd ../..
```

The consumer's `package.json` points to the local tarball under `javascript/artifacts/`. Installing the package only at the repository root does not populate this benchmark consumer.

Run benchmarks manually and keep generated results out of Git. CI does not run these benchmarks automatically. Large-scale and GPU measurements require a separate execution decision.

Record data, parameters, environment, and timing conditions with every result. Historical timing and memory records do not qualify a new source build. See [understanding performance](../docs/performance.md) for comparison guidance.
