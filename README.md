# rust-ga

This is a simple (but pretty flexible) bit-string genetic algorithm
in Rust so I can compare its performance to a similar system in
Clojure.

See [Planning.md](Planning.md) for info on where we're heading
and what we've accomplished.

# Optimized build
To build a fully optimized build (of the example) using the release-opt profile I'd reccomend using the following command:
```bash
RUSTFLAGS="-Zlocation-detail=none -Crelocation-model=static -Ctarget-cpu=native -Ztune-cpu=native" cargo +nightly build -Z build-std=std,panic_abort -Z build-std-features=panic_immediate_abort --target x86_64-unknown-linux-gnu --profile release-opt --example complex_regression
```

be aware that `-Ctarget-cpu=native -Ztune-cpu=native` can result in non-portable builds, so only do that for builds you don't intend to publish online.
Also adding `-Crelocation-model=static` can reduce binary size by a bit but it disables relocation so that might not be desireable.
