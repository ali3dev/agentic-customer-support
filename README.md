# Agentic Customer Support --- LLM Evaluation Foundation

A Python-based customer support evaluation foundation focused on prompt
engineering, structured output validation, behavior detection, and
reliable evaluation of AI-generated responses.

This project explores how to define expected AI behavior, detect unsafe
or unsupported responses, and evaluate outputs without treating
uncertain results as successful.

> **Project status:** Module 2 learning milestone --- Prompt Engineering
> & Structured Output. This is an evolving Grand Project that will be
> extended in later modules. It is not yet a fully integrated LLM
> application or autonomous customer support agent.

## 1. Problem Statement

Customer support AI systems must provide helpful responses while
avoiding unsupported claims.

For example, an assistant should not: - Claim a refund has been approved
without authorization. - Invent refund policies or delivery dates. -
Treat unverified customer statements as confirmed company policy. -
Pretend to have access to tracking information when it does not.

The purpose of this project is to establish a foundation for defining,
validating, and evaluating these behaviors.

## 2. Current Features

-   **Prompt engineering:** A customer support prompt with explicit
    instructions and safety constraints.
-   **Structured output validation:** Pydantic models validate response
    fields, allowed statuses, and strict Boolean values.
-   **Rule-based evaluation:** Regex patterns detect selected behaviors,
    including unsupported refund approval claims and invented refund
    periods.
-   **Pluggable judge interface:** A Python Protocol defines a contract
    for semantic behavior evaluation.
-   **Evaluation pipeline:** Expected and forbidden behaviors are
    evaluated independently.
-   **Uncertainty handling:** Unknown evaluation results are classified
    as `INCOMPLETE` rather than silently passing.
-   **Golden test cases:** JSON-based scenarios define expected and
    forbidden response behaviors.
-   **Automated tests:** Pytest verifies schema validation, detection
    rules, evaluation logic, and reporting.

## 3. Current Architecture

``` mermaid
flowchart TD
    A["Customer Input"] --> B["Support Prompt"]
    B --> C["Assistant Response (provided to evaluator)"]
    C --> D["Behavior Evaluator"]
    D --> E["Rule-Based Detection"]
    D --> F["Semantic Judge Interface"]
    E --> G["Evaluation Result"]
    F --> G
    H["Golden Test Cases"] --> D
    G --> I["PASS / FAIL / INCOMPLETE"]
    J["Pydantic Support Schema"] --> K["Structured Output Validation"]
```

**Important:** The current repository implements the evaluation and
validation components. It does not yet connect a live LLM provider to
the prompt or automatically generate assistant responses. The semantic
judge is currently an interface; tests use mock implementations.

## 4. Project Structure

``` text
agentic-customer-support/
├── prompts/
│   └── support_v1.txt
├── src/
│   ├── rules.py
│   ├── evaluator.py
│   ├── judge.py
│   ├── evaluation_runner.py
│   └── schemas/
│       └── support.py
├── tests/
│   ├── golden_cases.json
│   ├── test_rules.py
│   ├── test_evaluator.py
│   ├── test_judge.py
│   ├── test_evaluation_runner.py
│   └── test_support_schema.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## 5. Evaluation Design

The evaluator checks two categories:

**Expected behaviors** - Behaviors that should appear in an assistant
response. - Missing expected behaviors cause a failure.

**Forbidden behaviors** - Behaviors that must not appear. - Detected
forbidden behaviors cause a failure.

The evaluation system supports three outcomes:

  -----------------------------------------------------------------------
  Result                              Meaning
  ----------------------------------- -----------------------------------
  `PASS`                              All checks were verified
                                      successfully

  `FAIL`                              A required behavior was missing or
                                      a forbidden behavior was detected

  `INCOMPLETE`                        One or more behaviors could not be
                                      verified
  -----------------------------------------------------------------------

A key design decision is that uncertainty is not treated as success.

## 6. Technology Stack

-   Python 3.12+
-   Pydantic 2
-   Pytest
-   uv
-   Regular expressions
-   Python Protocols and dataclasses
-   JSON-based evaluation datasets

## 7. Getting Started

### Prerequisites

-   Python 3.12 or newer
-   uv package manager
-   Git

### Clone the repository

``` bash
git clone https://github.com/ali3dev/agentic-customer-support.git
cd agentic-customer-support
```

### Install dependencies

``` bash
uv sync
```

### Run the automated tests

``` bash
uv run pytest -q
```

**Recorded Module 2 baseline:** 28 tests passed during the project
review.

No LLM provider API key is required for the current test suite.

## 8. Engineering Decisions

### Strict structured output

Pydantic validates the response schema. `StrictBool` prevents strings
such as `"false"` from being silently accepted as actual Boolean values.

### Separation of evaluation responsibilities

Rule-based detection and semantic evaluation are separate components.
This makes it possible to introduce different judge implementations
without rewriting the main evaluator.

### Explicit uncertainty

An unavailable or uncertain judge result is represented as `unknown`.
The evaluator does not silently convert it into a successful result.

### Testable components

The evaluation system uses mock judges in automated tests. This allows
core application logic to be tested without depending on an external LLM
service.

## 9. Current Limitations

This is an evolving engineering project. The following capabilities are
not yet implemented:

-   Live LLM provider integration.
-   Automatic prompt execution.
-   End-to-end response generation and evaluation.
-   Production semantic judge implementation.
-   Robust JSON parsing and recovery from malformed model output.
-   Expanded evaluation datasets and regression benchmarks.
-   Production logging, latency tracking, and token-cost monitoring.
-   Tool calling, RAG, and autonomous agent workflows.

These capabilities are intended to be explored and integrated in later
learning modules as the same Grand Project evolves.

## 10. Project Roadmap

The same project will evolve through future engineering milestones:

-   [x] Prompt engineering foundation.
-   [x] Structured output schema.
-   [x] Rule-based behavior detection.
-   [x] Pluggable semantic judge contract.
-   [x] Evaluation pipeline and automated tests.
-   [ ] Live LLM integration.
-   [ ] Real semantic evaluation.
-   [ ] Tool and function calling.
-   [ ] Retrieval-Augmented Generation (RAG).
-   [ ] Agent orchestration and memory.
-   [ ] Evaluation benchmarks and observability.
-   [ ] Security, deployment, and production reliability.

## 11. Learning Objectives

This milestone demonstrates practical exploration of:

-   Prompt design as an engineering discipline.
-   Structured output validation.
-   Python interface design and separation of concerns.
-   Rule-based and semantic evaluation strategies.
-   Explicit handling of uncertainty.
-   Automated testing of AI application components.

## 12. Author

**Ali**

GitHub: https://github.com/ali3dev

This repository is part of an ongoing Agentic AI Engineering learning
journey, focused on building reliable, testable, and production-oriented
AI applications.

------------------------------------------------------------------------

*Built incrementally through hands-on engineering, testing, and
continuous improvement.*
