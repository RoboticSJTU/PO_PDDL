# PO-PDDL Language Specification

- **Status:** Current repository syntax, August 2026
- **File format:** PDDL-style S-expressions
- **Scope:** The PO-PDDL subset accepted by the parser, linter, code generator,
  and interactive runtime in this repository

## 1. Purpose and scope

PO-PDDL is a symbolic language for finite, relational partially observable
Markov decision processes. It retains typed PDDL action schemas and PPDDL
probabilistic effects, and adds three constructs needed for partially
observable planning:

- `:observables` declares the vocabulary of sensor reports;
- `:observation` defines state-conditioned observation distributions; and
- `:init-belief` defines a factorized probability distribution over initial
  states.

The current problem syntax also supports `:goal-reward`, which adds a scalar
reward when an action's successor state satisfies the goal.

This document specifies the supported language itself. Names and predicates
introduced by a particular learned domain are ordinary user-defined symbols
and are not part of PO-PDDL.

PO-PDDL is intentionally a practical subset of PDDL/PPDDL rather than a full
implementation of either standard. Section 9 lists the most important
boundaries of the current implementation.

## 2. Lexical conventions

PO-PDDL files are parenthesized S-expressions.

- Whitespace separates tokens.
- `(` and `)` are individual tokens.
- `;` starts a comment that continues to the end of the line.
- Variables begin with `?` by convention.
- Typed lists use PDDL notation: `a b - type`.
- Symbols and keywords should be lowercase. The current implementation treats
  symbol spelling as case-sensitive.
- A file must contain exactly one top-level `(define ...)` expression.

The built-in root type is `object`. An untyped declaration defaults to
`object`.

## 3. Files in a model

A runnable model consists of a domain and a problem:

```text
domain.pddl           Domain vocabulary, actions, and observation model
problem.pddl          Objects, simulator state, initial belief, and goal
```

The domain and problem names must agree through the problem's `:domain`
declaration.

## 4. Domain syntax

### 4.1 Top-level form

```pddl
(define (domain <domain-name>)
  (:requirements ...)
  (:types ...)
  (:constants ...)
  (:functions ...)
  (:predicates ...)
  (:observables ...)
  (:action ...)
  (:observation ...)
)
```

`:constants`, `:functions`, and `:observables` may be omitted when unused.
There may be any number of `:action` and `:observation` blocks.

### 4.2 Requirements

A representative declaration is:

```pddl
(:requirements
  :strips
  :typing
  :negative-preconditions
  :universal-preconditions
  :probabilistic-effects
  :conditional-effects
  :fluents
)
```

The section records the PDDL/PPDDL features used by a domain and improves
interoperability with external tooling. The current PO-PDDL parser does not use
`:requirements` as a feature gate; actual support is defined by the operators
listed in this document.

### 4.3 Types

`:types` defines a single-inheritance hierarchy rooted at `object`:

```pddl
(:types
  movable container - object
  fruit tool - movable
  drawer refrigerator - container
)
```

An argument declared as a parent type accepts objects of that type and all its
descendants. Objects and constants are included when quantified variables and
action parameters are grounded.

### 4.4 Constants

Domain constants are typed objects available in every problem:

```pddl
(:constants
  home - location
)
```

They may appear in predicates, actions, observation rules, and goals without
being repeated in `:objects`.

### 4.5 Predicates

`:predicates` declares Boolean state variables:

```pddl
(:predicates
  (hand-empty)
  (holding ?item - movable)
  (open ?container - container)
  (in ?item - movable ?container - container)
)
```

After grounding, each predicate atom is either true or false. State predicates
describe the latent world and are distinct from observables.

### 4.6 Observables

`:observables` declares Boolean sensor-report symbols:

```pddl
(:observables
  (obs-in ?item - movable ?container - container)
  (obs-open ?container - container)
  (obs-nothing)
)
```

An observation can report an observable as true, report it as false, or leave
it unreported. Therefore, an observation is a partial truth assignment over
ground observable atoms, not a world state.

`obs-nothing` is an optional reserved convention for an explicit empty sensor
signal. It is not inserted automatically. If it is emitted together with any
other positive observable, the runtime removes `obs-nothing` from the merged
observation.

### 4.7 Numeric reward function

The current runtime uses numeric effects for immediate reward accounting. The
portable convention is:

```pddl
(:functions
  (total-reward)
)
```

`total-reward` is not a general persistent numeric state fluent in the current
runtime. Numeric action effects contribute to the reward of that transition,
and the problem metric identifies the reward function to optimize.

## 5. Conditions

The same condition language is used in action preconditions, observation-rule
conditions, and goals.

```text
<condition> ::=
    (<predicate> <term>*)
  | (and <condition>*)
  | (or <condition>*)
  | (not <condition>)
  | (imply <condition> <condition>)
  | (= <term> <term>)
  | (forall (<typed-variable-list>) <condition>)
  | (all (<typed-variable-list>) <condition>)
  | (exists (<typed-variable-list>) <condition>)
```

Semantics:

| Form | Meaning |
|---|---|
| `(and c1 ... cn)` | Every child is true; `(and)` is true. |
| `(or c1 ... cn)` | At least one child is true; `(or)` is false. |
| `(not c)` | Boolean negation. |
| `(imply a b)` | Equivalent to `(or (not a) b)`. |
| `(= x y)` | Object/symbol equality. |
| `(forall (...) c)` | `c` is true for every compatible grounding. |
| `(all (...) c)` | Alias of `forall`. |
| `(exists (...) c)` | `c` is true for at least one compatible grounding. |

Predicates absent from a state are false under the closed-world assumption.
Quantifiers range over problem objects and domain constants whose declared
types are compatible with the variable type.

Current conditions do not support numeric comparisons or arithmetic
expressions.

## 6. Actions and transition semantics

### 6.1 Action schema

```pddl
(:action <action-name>
  :parameters (<typed-variable-list>)
  :precondition <condition>
  :effect <effect>
)
```

An omitted or empty precondition is treated as true. An action is applicable
when its grounded precondition is true in the current state.

### 6.2 Effect grammar

```text
<effect> ::=
    (<predicate> <term>*)
  | (not (<predicate> <term>*))
  | (and <effect>*)
  | (when <condition> <effect>)
  | (forall (<typed-variable-list>) <effect>)
  | (probabilistic <weight> <effect> <weight> <effect> ...)
  | (increase (<reward-function>) <number>)
  | (decrease (<reward-function>) <number>)
  | (assign (<reward-function>) <number>)
```

A positive atom sets a grounded predicate to true. A negated atom sets it to
false. `(and)` is a no-op and is useful as an explicit failure or no-change
outcome.

The guard of a `when` effect is evaluated in the pre-action state. A `forall`
effect applies its body to every type-compatible grounding.

Do not assign both true and false to the same grounded predicate in one effect
outcome; such conflicting effects are not portable across execution backends.

### 6.3 Probabilistic effects

```pddl
(probabilistic
  0.85 (and (holding ?item) (not (hand-empty)))
  0.15 (and)
)
```

Each branch weight must be non-negative and at least one branch must have a
positive weight. The runtime samples branches proportionally to their weights.
Authors should make weights sum to `1.0` for readability and interoperability.

There is **no implicit residual no-op outcome** when weights sum to less than
`1.0`. The listed weights are normalized. Model a no-change outcome explicitly
with a branch such as `0.15 (and)`.

Conjunctive effects may combine deterministic, conditional, and probabilistic
components. Nested probabilistic effects are supported.

### 6.4 Reward effects

```pddl
(and
  (decrease (total-reward) 2.5)
  (probabilistic
    0.9 (open ?container)
    0.1 (and)
  )
)
```

`increase`, `decrease`, and `assign` update the immediate transition reward.
Action costs are normally encoded with `decrease`. The stable syntax uses a
numeric literal as the update amount.

## 7. Observation model

### 7.1 Observation rule

```pddl
(:observation <rule-name>
  :parameters (<typed-variable-list>)
  :condition <condition>
  :distribution <observation-distribution>
)
```

The distribution grammar is:

```text
<observation-distribution> ::=
    (<observable> <term>*)
  | (not (<observable> <term>*))
  | (and <observation-distribution>*)
  | (probabilistic
      <weight> <observation-distribution>
      <weight> <observation-distribution>
      ...)
```

Example:

```pddl
(:observation observe-item-present
  :parameters (?item - movable ?container - container)
  :condition (and (open ?container) (in ?item ?container))
  :distribution (probabilistic
    0.9 (obs-in ?item ?container)
    0.1 (not (obs-in ?item ?container))
  )
)
```

`(obs-in ...)` explicitly reports the observable as true.
`(not (obs-in ...))` explicitly reports it as false. An observable omitted from
the sampled distribution branch is unreported and supplies no evidence by
itself.

### 7.2 Generative semantics

After an action produces successor state `s'`:

1. Ground every observation rule over type-compatible objects and constants.
2. Evaluate each grounded rule's `:condition` in `s'`.
3. For every enabled rule, independently sample one branch from its
   `:distribution`, proportionally to the listed non-negative weights.
4. Merge the sampled partial observable assignments into one compound
   observation.

As with action effects, authors should make each distribution sum to `1.0`,
but the runtime normalizes the listed weights and does not add a residual
branch.

Overlapping rules should not emit contradictory truth values for the same
ground observable in the same state. Use mutually exclusive conditions or a
single combined rule when reports are correlated.

If no rule is enabled, the observation is empty. To represent a named empty
signal, define and explicitly emit `(obs-nothing)`.

## 8. Problem syntax

### 8.1 Top-level form

```pddl
(define (problem <problem-name>)
  (:domain <domain-name>)
  (:objects ...)
  (:init ...)
  (:init-belief ...)
  (:goal <condition>)
  (:goal-reward <number>)
  (:metric maximize (<reward-function>))
)
```

`:objects` may be empty. `:goal-reward` is optional and defaults to `0.0`.

### 8.2 Objects

```pddl
(:objects
  apple - fruit
  source target - drawer
)
```

Every object type must be declared in the domain. Object names and domain
constant names should be unique.

### 8.3 Simulator initial state

`:init` specifies the true initial state used by simulation or interactive
execution:

```pddl
(:init
  (hand-empty)
  (in apple source)
  (not (open source))
)
```

Positive and explicitly negated ground predicates are accepted. Omitted
predicates are false. In a partially observable problem, this state may contain
facts that are uncertain in `:init-belief`; the planner does not receive
`:init` as its belief.

General numeric fluent initialization is not part of the current runtime
semantics. Transition reward starts at zero and is produced by action effects.

### 8.4 Factorized initial belief

`:init-belief` describes what the planner knows before acting. Four entry forms
are supported.

#### Certain facts

```pddl
(:init-belief
  (hand-empty)
  (not (holding apple))
)
```

A bare atom is certainly true. A top-level negated atom is certainly false.

#### Independent Bernoulli fact

```pddl
(prob (open source) 0.25)
```

The grounded predicate is true with probability `0.25` and false with
probability `0.75`. Decimal and rational probabilities such as `1/4` are
accepted.

#### Correlated factor

```pddl
(joint 0.5 (and
  (in apple source)
  (not (in apple target))))
(joint 0.5 (and
  (not (in apple source))
  (in apple target)))
```

Consecutive `joint` entries with the same predicate scope form one mutually
exclusive factor. Their probabilities must sum to `1.0` within a tolerance of
`1e-3`. Predicates in the factor scope that are not true in a case are false in
that case; writing the negative literals explicitly makes the scope and intent
unambiguous.

#### Uniform correlated factor

```pddl
(oneof
  (and (in apple source) (not (in apple target)))
  (and (not (in apple source)) (in apple target)))
```

`oneof` is shorthand for equally weighted mutually exclusive assignments in
`:init-belief` only.

#### Factor composition

Certain facts and uncertain factor scopes must not overlap. Two uncertain
factors must also have disjoint scopes. Distinct factors are independent, and
the initial belief is their exact Cartesian product. Cases yielding the same
world state are combined by summing their weights.

Every grounded predicate omitted from certain facts and all factor scopes is
false by default. For clarity, generated problems commonly emit deterministic
facts as `(prob (<atom>) 1.0)` or `(prob (<atom>) 0.0)`; these are equivalent to
bare positive or negative certain facts.

### 8.5 Goal

`:goal` contains one condition expression:

```pddl
(:goal
  (and
    (in apple target)
    (not (open target))
  )
)
```

The goal is evaluated on a concrete latent state. The same logical operators
listed in Section 5 are supported.

### 8.6 Goal reward and metric

```pddl
(:goal-reward 100)
(:metric maximize (total-reward))
```

Whenever an ordinary action's successor state satisfies the goal, the runtime
adds `:goal-reward` to that transition's reward. Runtime configurations may
override this value for an experiment.

The metric direction may be `maximize` or `minimize`. PO-PDDL experiments in
this repository normally maximize `total-reward`, with action execution time
or cost encoded as negative reward.

## 9. Current supported subset

The following constructs from broader PDDL/PPDDL dialects are not part of the
current stable PO-PDDL syntax:

- derived predicates and axioms;
- state-constraint or state-invariant sections;
- numeric comparisons in conditions and goals;
- general persistent numeric fluents;
- `oneof` or `mutex` in conditions and action effects;
- nondeterministic effects outside `probabilistic`;
- existential effects; and
- an implicit residual branch for incomplete probability sums.

Unknown top-level sections may be ignored by the lightweight parser. They
should not be relied upon: only sections and operators documented here are
validated and compiled into runtime behavior.

## 10. Formal model

A domain/problem pair denotes a POMDP

`M = <S, A, T, Omega, O, R, b0, G>`:

| Component | PO-PDDL source |
|---|---|
| `S` | Truth assignments to all grounded `:predicates` |
| `A` | Type-correct groundings of `:action` schemas |
| `T(s' | s, a)` | Deterministic, conditional, quantified, and probabilistic action effects |
| `Omega` | Partial truth assignments to grounded `:observables` |
| `O(o | s', a)` | Enabled grounded `:observation` rules and their distributions |
| `R(s, a, s')` | Numeric reward effects plus optional goal reward |
| `b0` | Exact factorized distribution in `:init-belief` |
| `G` | State formula in `:goal` |

The planning discount factor, horizon, particle approximation, and stopping
criterion are runtime parameters rather than language constructs.

## 11. Complete example

### Domain

```pddl
(define (domain hidden-item)
  (:requirements
    :strips :typing :negative-preconditions
    :universal-preconditions :probabilistic-effects
    :conditional-effects :fluents)

  (:types
    item container - object)

  (:predicates
    (hand-empty)
    (holding ?item - item)
    (open ?container - container)
    (in ?item - item ?container - container))

  (:observables
    (obs-in ?item - item ?container - container))

  (:functions
    (total-reward))

  (:action open-container
    :parameters (?container - container)
    :precondition (and (hand-empty) (not (open ?container)))
    :effect (and
      (decrease (total-reward) 1)
      (open ?container)))

  (:action pick-item
    :parameters (?item - item ?container - container)
    :precondition (and
      (hand-empty)
      (open ?container)
      (in ?item ?container))
    :effect (and
      (decrease (total-reward) 2)
      (probabilistic
        0.9 (and
          (holding ?item)
          (not (hand-empty))
          (not (in ?item ?container)))
        0.1 (and))))

  (:action place-item
    :parameters (?item - item ?container - container)
    :precondition (and
      (holding ?item)
      (open ?container))
    :effect (and
      (decrease (total-reward) 2)
      (in ?item ?container)
      (hand-empty)
      (not (holding ?item))))

  (:observation item-present
    :parameters (?item - item ?container - container)
    :condition (and
      (open ?container)
      (in ?item ?container))
    :distribution (probabilistic
      0.9 (obs-in ?item ?container)
      0.1 (not (obs-in ?item ?container))))

  (:observation item-absent
    :parameters (?item - item ?container - container)
    :condition (and
      (open ?container)
      (not (in ?item ?container)))
    :distribution (probabilistic
      0.1 (obs-in ?item ?container)
      0.9 (not (obs-in ?item ?container))))
)
```

### Problem

```pddl
(define (problem move-hidden-item)
  (:domain hidden-item)

  (:objects
    package - item
    source target - container)

  ; Ground truth for simulation: the package starts in source.
  (:init
    (hand-empty)
    (in package source))

  ; The planner initially assigns equal probability to either container.
  (:init-belief
    (prob (hand-empty) 1.0)
    (prob (holding package) 0.0)
    (prob (open source) 0.0)
    (prob (open target) 0.0)
    (joint 0.5 (and
      (in package source)
      (not (in package target))))
    (joint 0.5 (and
      (not (in package source))
      (in package target))))

  (:goal
    (in package target))

  (:goal-reward 100)
  (:metric maximize (total-reward))
)
```

## 12. Validation with the repository parser

Use the parser and cross-file linter when authoring a model:

```python
from pathlib import Path

from po_pddl.core.linter import lint_texts

domain_text = Path("domain.pddl").read_text()
problem_text = Path("problem.pddl").read_text()
result = lint_texts(domain_text, problem_text)

for diagnostic in result.diagnostics:
    print(diagnostic.severity, diagnostic.code, diagnostic.message)

if not result.ok:
    raise SystemExit("PO-PDDL validation failed")
```

The interactive runtime also supports `po-pddl-run-terminal --validate-only`
to parse and compile a complete planning bundle without entering the execution
loop.
