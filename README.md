# ModelInquisitor - Claims-based BPMN Translation Checker

ModelInquisitor checks whether an mCRL2 translation preserves key semantic facts from the source BPMN model. It does not try to prove full equivalence between BPMN and mCRL2. Instead, it extracts semantic invariants as Claims, renders them as modal mu-calculus formulas, and verifies those formulas against the translated model.

The default naming strategy is compatible with `third-party/bpmn2mcrl2`, but that compatibility lives behind a replaceable strategy interface. The parser, Claim extractors, formula generator, and runner do not depend directly on one concrete translator.

## Architecture

```text
ModelInquisitor/
├── core/        # BPMN model, Claims, and graph algorithms
├── parsers/     # BPMN XML parsing and NetworkX graph export
├── extractors/  # Semantic Claim extractors
├── generators/  # Claim -> MCF formula generation
├── strategies/  # Replaceable BPMN -> mCRL2 naming strategies
└── runners/     # mCRL2 toolchain orchestration
```

## Claims

All claim kinds use `prefix::claim_name`, with prefixes limited to `soundness`, `flow`, `concurrency`, and `interaction`.

- **Soundness claims**: `soundness::deadlock_freedom`, `soundness::action_preservation`, `soundness::end_event_preservation`, and `soundness::bounded_unfolding_soundness` check termination reachability, observable action preservation, per-end-event preservation, and bounded loop unfolding.
- **Flow claims**: `flow::causality`, `flow::mutex`, `flow::necessary_response`, `flow::exclusive_branch_reachability`, `flow::exclusive_branch_mutex`, `flow::event_based_first_wins`, `flow::event_based_branch_reachability`, `flow::escape_possibility`, and `flow::no_forced_starvation` check control-flow ordering, exclusivity, reachability, response, and loop escape properties.
- **Concurrency claims**: `concurrency::no_artificial_ordering`, `concurrency::branch_order_preservation`, `concurrency::branch_co_occurrence`, `concurrency::no_early_join`, `concurrency::join_reachable_after_all_branches`, and `concurrency::exactly_once_branch_completion_before_join` check parallel interleavings and join behavior.
- **Interaction claims**: `interaction::rendezvous_visibility`, `interaction::rendezvous_causality`, `interaction::conversation_order_preservation`, and `interaction::no_post_resolution_chatter` check message-flow synchronization, participant-side causality, conversation order, and post-resolution chatter.

Claim formulas mention only actions that represent BPMN-visible semantics. Translator-internal helper actions such as raw `s_*`/`r_*` message endpoints and gateway synchronization actions are treated as implementation detail rather than semantic claim targets. For a BPMN message flow, the communicated `c_*` action is the semantic representative of the message exchange.

## Command-line Verification

Install dependencies with `uv`:

```powershell
uv sync --dev
```

Verify a BPMN/mCRL2 pair:

```powershell
uv run python main.py tests/input/spec.bpmn tests/input/spec.mcrl2
```

Keep intermediate artifacts and print generated formulas:

```powershell
uv run python main.py tests/input/spec.bpmn tests/input/spec.mcrl2 --work-dir .verify-artifacts --show-formulas
```

The runner calls the real mCRL2 toolchain:

```text
mcrl22lps -> lps2pbes -> pbes2bool
```

The default UI prints a compact result table and a grouped explanation of what was checked. Detailed panels are shown only for failures or when `--show-formulas` is used.

Exit codes:

- `0`: all Claims were verified as true.
- `1`: at least one Claim was verified as false.
- `2`: an input file was missing.
- `3`: model conversion, formula generation, or solving failed.
