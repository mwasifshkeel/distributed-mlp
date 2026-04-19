# Distributed MLP Training

CS-347 Parallel & Distributed Computing | NUST SEECS

## Overview
This repository contains the implementation of a distributed multi-layer perceptron (MLP) training pipeline for the Food-101 dataset.

Project focus:
- Sequential baseline for correctness and reference timing
- Intra-node multithreading on workers
- Distributed asynchronous SGD over TCP sockets
- Benchmarking, scaling analysis, and Amdahl's Law validation
- Communication optimization using gradient compression

## Planned Stack
- Java 17
- Maven 3.9+
- TCP sockets (`java.net.Socket`)
- Concurrency primitives from `java.util.concurrent`

## Repository Structure
```text
distributed-mlp/
├── pom.xml
├── README.md
├── run.sh
├── src/main/java/com/distributed/mlp/
└── results/
```

## Build
```bash
mvn package
```


## Team
- [Muhammad Wasif Shakeel](https://github.com/mwasifshkeel)
- [Qasim Ahmed](https://github.com/qasimahmed06)
- [Muhammad Hanzala Bin Rehan](https://github.com/Hanzalarehan)