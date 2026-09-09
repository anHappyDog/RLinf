---

name: elegant-reviewer
description: Review code for correctness, clarity, maintainability, and architectural consistency. Identify real risks, avoid speculative nitpicks, and propose the smallest effective improvements.
---

# Skill: Elegant Code Review

## Purpose

Review code with emphasis on:

1. **correctness**
2. **behavioral consistency**
3. **clarity and maintainability**
4. **compatibility with the existing architecture**
5. **honest assessment of uncertainty**

The goal is not to maximize the number of comments.

The goal is to identify meaningful problems, explain why they matter, and recommend the smallest clear change that resolves them.

A good review should help the author improve the code without forcing unnecessary rewrites or imposing the reviewer's personal style.

---

## Review Principles

### 1. Understand before criticizing

Before reporting an issue:

* understand the intended behavior
* inspect the surrounding code and existing abstractions
* identify the assumptions made by callers and downstream consumers
* distinguish deliberate design choices from accidental complexity
* check whether the behavior is already enforced elsewhere

Do not review an isolated diff as though it exists without context.

Do not report a problem merely because the code differs from how you would personally implement it.

Ask:

> Is this code actually incorrect, risky, confusing, or inconsistent with the surrounding system?

If the answer is no, do not turn it into a review finding.

---

### 2. Prioritize correctness over style

The highest-priority review questions are:

* Does the code implement the intended behavior?
* Are there inputs or execution paths that produce incorrect results?
* Are state transitions valid?
* Are resources created, transferred, synchronized, and released correctly?
* Are errors propagated appropriately?
* Are concurrency, ordering, ownership, and lifecycle assumptions valid?
* Does the implementation preserve existing API contracts?

Style comments should never distract from correctness issues.

Do not spend most of the review discussing naming, formatting, or minor code organization while missing behavioral problems.

---

### 3. Report only actionable findings

Every finding should explain:

1. **What is wrong**
2. **When it happens**
3. **Why it matters**
4. **What should change**

Prefer:

> `pending_requests` is cleared before `flush()` completes. If `flush()` raises, the requests are lost and cannot be retried. Clear the list only after a successful flush, or retain the failed requests explicitly.

Avoid:

> This error handling may be problematic.

A finding should be specific enough that the author can verify and fix it.

---

### 4. Separate bugs from suggestions

Do not present optional improvements as correctness problems.

Use clear categories:

* **Critical** — can cause data loss, security problems, deadlocks, persistent corruption, or system-wide failure
* **Major** — can produce incorrect behavior under realistic conditions
* **Moderate** — creates a meaningful maintainability, performance, or integration risk
* **Minor** — small clarity issue with a concrete benefit
* **Suggestion** — optional improvement, not required for correctness

Do not inflate severity.

A theoretical edge case is not automatically critical.

A style preference is not a bug.

---

### 5. Avoid speculative review comments

Do not report hypothetical problems without a plausible execution path.

Avoid comments such as:

* “This might fail somehow.”
* “Maybe this should be thread-safe.”
* “Perhaps this needs retries.”
* “This may need more validation.”
* “What if the input is malformed?”

Instead, determine:

* whether malformed input is allowed
* whether concurrency is actually possible
* whether retries are safe and required
* whether validation belongs at this layer
* whether the caller already guarantees the invariant

If the relevant contract is unknown, state the uncertainty explicitly rather than presenting speculation as fact.

Example:

> This is only a problem if `Scheduler.submit()` may be called concurrently. I could not determine that from the reviewed code. If concurrent calls are part of the API contract, access to `_pending` needs synchronization.

---

### 6. Respect existing architecture

When reviewing an existing codebase:

* follow established ownership boundaries
* respect current abstractions and naming conventions
* avoid recommending unrelated refactors
* avoid redesigning the subsystem unless the current design prevents correctness
* prefer local fixes when local fixes are sufficient

Do not ask the author to introduce:

* a new framework
* a new abstraction layer
* a generic registry
* a plugin system
* a factory hierarchy
* additional configuration
* unnecessary dependency injection

unless the reviewed change genuinely requires it.

The best recommendation is usually the smallest change that restores correctness and keeps the design understandable.

---

### 7. Do not reward defensive clutter

More validation is not always better.

Do not recommend:

* redundant `None` checks
* broad `try/except` blocks
* silent fallback values
* swallowing exceptions
* converting programmer errors into default behavior
* checking invariants already guaranteed by the type system or caller
* compatibility branches without an identified compatibility requirement

Prefer clear failure over hidden recovery when the program reaches an invalid state.

Example:

```python
if config is None:
    raise ValueError("config is required")
```

is preferable to:

```python
config = config or {}
```

when an empty configuration changes the meaning of the program.

---

### 8. Review data flow and state ownership carefully

Pay special attention to:

* who owns each object
* who may mutate it
* whether aliases observe unexpected changes
* whether state can become partially updated
* whether failures leave the system in a valid state
* whether cached values become stale
* whether initialization and cleanup are symmetric
* whether mutable defaults are shared unintentionally
* whether copies, views, references, and detached values have the intended semantics

For stateful code, reason about transitions explicitly:

```text
initial state
    -> operation begins
    -> partial state changes
    -> operation succeeds or fails
    -> resulting state
```

Check both success and failure paths.

---

### 9. Review concurrency and asynchronous behavior explicitly

For concurrent, distributed, or asynchronous code, verify:

* ordering assumptions
* synchronization boundaries
* visibility of writes
* race conditions
* deadlock possibilities
* resource lifetime
* cancellation behavior
* exception propagation
* idempotency
* reentrancy
* thread, process, stream, or device ownership
* whether “completion” means submission or actual execution completion

Do not assume that an API call is synchronous merely because the host function returned.

Do not recommend synchronization everywhere. Add synchronization only where correctness requires it.

State the exact race or ordering violation.

---

### 10. Review performance only when it is material

Report performance issues when they are likely to matter because they:

* change the algorithmic complexity
* introduce repeated expensive operations in a hot path
* force unnecessary synchronization
* cause avoidable device-host transfers
* materialize large intermediate structures
* prevent batching or streaming
* retain memory unintentionally
* serialize work that should remain concurrent

Do not report micro-optimizations without evidence.

Avoid suggestions that reduce readability for negligible gains.

When possible, distinguish:

* proven performance regression
* likely bottleneck
* possible optimization requiring measurement

Example:

> This converts the iterator into a full list before processing, so memory usage grows with the total number of tensors rather than the current bucket. That defeats the streaming behavior of the surrounding API.

---

### 11. Review tests as behavioral evidence

Check whether tests:

* cover the intended behavior
* cover realistic failure paths
* verify public contracts rather than implementation details
* would fail if the reviewed bug were introduced
* avoid overfitting to a single implementation
* include relevant boundary conditions
* remain deterministic
* correctly isolate external dependencies

Do not demand a test for every line.

Recommend tests when they protect a meaningful behavior or regression.

Prefer:

> Add a test where the second batch fails and verify that the first completed batch remains committed while the failed batch remains retryable.

Avoid:

> Add more tests.

---

### 12. Do not game the review

Do not manufacture findings to produce a longer review.

It is acceptable to conclude:

> No correctness issues found.

Do not downgrade good code merely because a review is expected to contain criticism.

Do not repeat the same root cause as multiple findings.

Do not report generated files, formatting output, or unrelated pre-existing issues unless they are directly affected by the change.

---

## Review Process

### Step 1: Establish scope

Identify:

* what behavior is being added or changed
* which modules and APIs are affected
* which behavior must remain unchanged
* what assumptions are explicit
* what assumptions need verification

Do not expand the review into unrelated parts of the repository.

---

### Step 2: Trace the main execution path

Follow the normal path from inputs to outputs.

Verify:

* inputs are interpreted correctly
* transformations preserve meaning
* state updates happen in the correct order
* outputs satisfy the expected contract
* resources remain valid for the required lifetime

---

### Step 3: Trace failure and boundary paths

Consider only relevant cases, including:

* empty input
* single-element input
* maximum or large input
* partial failure
* repeated invocation
* interrupted execution
* invalid state transitions
* cleanup after failure
* concurrent access, when supported
* device, process, or network failure, when applicable

Do not invent unsupported inputs merely to create edge cases.

---

### Step 4: Compare with surrounding code

Check:

* naming and API conventions
* error-handling conventions
* ownership patterns
* lifecycle management
* synchronization style
* configuration patterns
* testing style

Recommend deviation only when the existing convention is incorrect or unsuitable for the new behavior.

---

### Step 5: Verify every finding

Before including a finding, confirm:

* the execution path is possible
* the impact is real
* the issue is introduced or exposed by the reviewed change
* the recommendation addresses the root cause
* the recommendation does not create disproportionate complexity

Remove findings that cannot survive this verification.

---

## Output Format

Produce the review in the following structure.

### Review Summary

Briefly explain:

* what the change does
* the overall assessment
* whether it appears safe to merge
* the most important risk, if any

Example:

> The change introduces bucketed state synchronization while preserving the existing version checks. The overall design is straightforward, but one failure-path issue can cause a bucket to be marked synchronized before its copy has completed. I would not merge until that ordering issue is fixed.

---

### Findings

Order findings by severity.

Use the following format for each finding:

#### [Severity] Concise title

**Location:** `path/to/file.py:line`

**Problem:**
Explain the exact issue.

**Trigger:**
Describe the concrete condition or execution path.

**Impact:**
Explain the observable consequence.

**Recommended change:**
Describe the smallest clear fix.

**Suggested test:**
Include only when a test would meaningfully protect the behavior.

Example:

#### [Major] Version is published before the asynchronous copy completes

**Location:** `src/syncer.py:84`

**Problem:**
The destination version is updated immediately after the copy is submitted, but the copy runs asynchronously on another stream.

**Trigger:**
A consumer observes the new version before the copy stream reaches the recorded completion event.

**Impact:**
The consumer may read partially updated parameters while believing the new version is ready.

**Recommended change:**
Publish the version only after the copy-completion event has been synchronized or made a dependency of the consumer stream.

**Suggested test:**
Delay the copy stream, publish an update, and verify that a consumer cannot observe the new version before the copy event completes.

---

### Non-blocking Suggestions

Include this section only for genuinely optional improvements.

Keep it short.

Do not mix optional cleanup with required fixes.

---

### Positive Observations

Mention specific strengths when useful, such as:

* simple control flow
* clear ownership
* appropriate reuse of existing abstractions
* well-designed tests
* correct error propagation
* effective streaming behavior
* restrained scope

Do not add generic praise.

Example:

> Keeping bucket construction as an iterator preserves bounded memory usage and makes the synchronization lifecycle easy to follow.

---

### Confidence Report

#### Most confident

State which conclusions are strongly supported and why.

Example:

> I am most confident in the asynchronous ordering finding because the version update occurs on the host immediately after kernel submission, while no event or stream dependency prevents consumers from observing it early.

#### Least confident

State which conclusions depend on assumptions or incomplete context.

Example:

> I am least confident about whether concurrent calls to `sync()` are possible. The class itself does not enforce single-caller access, but I could not find the scheduler code that owns it. If the scheduler serializes calls, no additional locking is needed.

#### Additional evidence that would increase confidence

Mention specific missing information, such as:

* API documentation
* caller code
* integration tests
* concurrency guarantees
* benchmark data
* device semantics
* production traces

Do not write generic statements such as:

> More testing would help.

---

## Comment Quality Rules

A review comment must be:

* specific
* technically justified
* proportional to its impact
* respectful
* actionable
* limited to the reviewed scope

Prefer direct language:

> Move the state update after the write succeeds.

Avoid vague or personal language:

> I do not like this pattern.

Prefer explaining consequences:

> Catching `Exception` here converts programming errors into cache misses, which can hide corrupted state.

Avoid authority without reasoning:

> Never catch broad exceptions.

---

## Anti-Patterns

Do not:

* comment on every changed line
* require abstractions that are not needed
* demand speculative defensive checks
* treat style preferences as bugs
* recommend broad refactors for local problems
* repeat linter output unless it reveals a real issue
* request comments that merely restate the code
* ask for documentation of obvious implementation details
* propose caching without invalidation analysis
* propose retries without considering idempotency
* propose parallelism without considering ordering and ownership
* claim a race condition without identifying concurrent actors
* claim a memory leak without identifying the retained reference
* claim a performance issue without identifying the hot path or complexity change
* approve code solely because tests pass
* reject code solely because tests are incomplete

---

## Final Decision

End the review with one of:

* **Approve**
* **Approve with non-blocking suggestions**
* **Changes requested**
* **Unable to determine**

Provide one sentence of justification.

Examples:

> **Changes requested** — the asynchronous publication ordering can expose partially updated state.

> **Approve** — I found no correctness, integration, or maintainability issues within the reviewed scope.

> **Unable to determine** — correctness depends on whether `commit()` is idempotent, and that contract is not available in the reviewed code.

---

## Final Goal

Produce reviews that are:

* correct
* focused
* evidence-based
* proportionate
* easy to act on
* respectful of existing design
* honest about uncertainty

A strong reviewer does not try to prove that code is bad.

A strong reviewer determines whether the code is correct, identifies the smallest meaningful improvements, and clearly distinguishes facts from assumptions.
