Implementation of Goldilocks and its extension fields
------

This repo implements 
- Goldilocks Field mod `2^64 - 2^32 + 1`
- Goldilocks quadratic extension over `x^2 - 7`
- Goldilocks cubic extension over `x^3 - x - 1`

Traits are compatible with `ff 0.13.0`.

### Benchmark
```
cargo bench
```