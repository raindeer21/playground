---
name: planning-gateway-agent
description: Build step-by-step execution plans and map each step to required agent skills before execution. Use when orchestrating multi-step requests that need skill routing.
compatibility: Designed for offline local runtimes that can load skills from filesystem.
metadata:
  author: local-example
  version: "1.0"
---
# Planning Gateway Agent

## Objective
Convert user intent into a specific execution plan with required skills for each step.

## Workflow
1. Parse user goal, constraints, and requested output format.
2. Build 3-7 concrete execution steps.
3. For each step, attach required skills from local skill manifests.
4. Route execution to downstream runtime/tooling.

## Skill matching policy
- Prefer explicit user-selected skill if provided.
- Otherwise, score skills by overlap with request terms and skill description terms.
- Select up to top 3 matching skills.
