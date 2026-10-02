# Examples

Build and install the local packages first; see [getting started](../docs/getting-started.md).

- [`python/all_methods.py`](python/all_methods.py): small public-API examples using an installed wheel
- [`python/som_variants.py`](python/som_variants.py): online SOM and BatchSOM
- [`javascript/realtime.js`](javascript/realtime.js): incremental sessions and Worker updates
- [`javascript/rmcm.js`](javascript/rmcm.js): RMCM usage
- [`browser/`](browser/): a Worker demo and browser test page

From the repository root:

```sh
.venv/bin/python -I -B examples/python/all_methods.py
node examples/browser/serve.js
```

Open http://localhost:8765/ in a browser. The development server listens only on `127.0.0.1`. Open http://localhost:8765/examples/browser/test.html for the browser test page.

The browser examples import repository source directly. To verify installed distribution artifacts, use the package checks described in [development and verification](../docs/contributing.md).
