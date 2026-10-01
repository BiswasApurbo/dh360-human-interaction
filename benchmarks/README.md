# Benchmarks

This directory contains benchmarking suites and performance profiling tools for the Digital Health 360 Human Interaction module.

## Objectives
- Track latency across individual pipeline stages:
  - Face detection latency (ms)
  - Face tracking & quality gate latency (ms)
  - Face embedding extraction latency (ms)
  - Local gallery vector search latency (ms)
- Verify end-to-end throughput against the hardware requirement:
  - **Raspberry Pi Edge Target**: `> 15 FPS` sustained real-time vision processing.
  - **MacBook Development Target**: Baseline reference profiling and regression prevention.
