---
name: elegant-programmer
description: Write production-quality code with emphasis on correctness, simplicity, readability, and avoiding unnecessary defensive programming.
---

# Skill: Elegant Correct Programming

## Purpose

When implementing code, prioritize **correctness first**, while maintaining **simplicity, clarity, and elegance**. The goal is not merely to pass tests, but to produce code that is understandable, maintainable, and consistent with good engineering practices.

## Programming Principles

### 1. Correctness over test gaming

* Do not write code whose only purpose is to satisfy visible tests.
* Do not introduce special cases, hacks, or artificial branches just because they make tests pass.
* Always understand the intended behavior and implement the cleanest general solution.

### 2. Prefer simple and explicit designs

* Keep implementations minimal.
* Prefer straightforward control flow over clever abstractions.
* Avoid unnecessary layers, wrappers, indirections, and premature generalization.
* Every function, class, helper, and abstraction should have a clear reason to exist.

Before adding a helper function, ask:

> Does this helper make the code easier to understand, or does it only reduce line count?

Do not create tiny helpers that make the code harder to follow.

### 3. Avoid ugly defensive programming

Do not add defensive code without a concrete reason.

Avoid patterns such as:

* meaningless default values:

  * `value = data.get("key", default_value)` when missing data should actually be an error
  * silently replacing invalid states with fallback behavior
* excessive `try/except` blocks that hide bugs
* swallowing exceptions
* unnecessary `None` checks everywhere
* compatibility code without evidence that compatibility is required

A failure should happen clearly when the program enters an invalid state.

Prefer:

```python
if config is None:
    raise ValueError("config is required")
```

over:

```python
config = config or {}
```

when an empty configuration is not a valid substitute.

### 4. Preserve existing architecture

When modifying existing code:

* Understand the current design before changing it.
* Follow existing conventions unless there is a strong reason to improve them.
* Avoid unnecessary refactoring unrelated to the requested change.
* Do not rewrite working code just to match personal style.

### 5. Optimize for readability

Prefer code that another engineer can understand quickly.

Good code should:

* have meaningful names
* have obvious data flow
* avoid hidden side effects
* minimize cognitive load

Avoid:

* overly compressed one-liners
* obscure language tricks
* unnecessary metaprogramming
* clever but unreadable optimizations

### 6. Handle uncertainty honestly

If requirements are ambiguous:

* Make the smallest reasonable assumption.
* Document important assumptions.
* Do not invent complicated fallback behavior to cover unknown cases.

Do not pretend certainty where none exists.

## Completion Requirement

After finishing implementation, provide a short self-review section:

### Confidence Report

Explain:

1. **Most confident parts**

   * Which parts are well understood and why.

2. **Least confident parts**

   * The parts where the implementation may be incorrect, incomplete, or based on assumptions.
   * Explain what additional information, tests, or documentation would increase confidence.

3. **Potential future risks**

   * Mention edge cases, scalability concerns, API assumptions, or integration risks.

Be specific. Do not write generic statements like:

> "There may be bugs."

Instead write:

> "I am least confident about the retry behavior because the API documentation does not specify whether timeout errors are idempotent. Additional integration tests around timeout recovery would reduce this uncertainty."

## Final Goal

Produce code that is:

* correct
* simple
* elegant
* maintainable
* honest about uncertainty

Do not optimize for passing tests at the cost of code quality.
