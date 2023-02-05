#!/bin/bash

cargo build --release
gcc -l torch -l torch_cpu -l m -l c10 -l gflags -l stdc++ -l bz2 test.c ./target/release/libravensaid.a
