# Understanding performance

Runtime depends on data shape, backend, initialization, thread count, warmup, JIT compilation, and WASM availability. A small smoke test does not establish a general speed or quality guarantee.

When comparing implementations:

- Use the same data, parameters, initialization, and stopping criteria
- Record the runtime, hardware, thread count, and selected backend
- Separate cold startup and compilation from warmed execution
- Compare output quality as well as elapsed time
- Account for data preparation, result copies, and Worker communication when measuring an application

Numerical recovery paths can be slower on extreme inputs. Read the [numerical contracts and limits](numerics.md) before interpreting timing differences.

The [benchmark guide](../benchmarks/README.md) describes the available manual drivers. Historical investigations are kept in a separate [research archive](performance/README.md); their measurements apply to the recorded versions and environments, not automatically to a new build.
