# Validation Notes

The source release provides lightweight offline checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/check_behavior_equivalence.py --self-test
PYTHONDONTWRITEBYTECODE=1 python scripts/check_method.py
PYTHONDONTWRITEBYTECODE=1 python scripts/check_release.py
```

The runtime check runs Raw, full PoS, both ablations, and diagnostic PoS with fixed model
responses and a mock environment. It exercises actual action transitions and
recovery without calling an API or generating experiment output files. The
method check covers the health equations, classification, independent confidence,
and recovery refresh/release. The package check scans for generated results,
private paths, credentials, and invalid Python syntax. These are not full
benchmark evaluations.

To compare behavior against a separately retained source tree:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/check_behavior_equivalence.py --reference /path/to/reference/source
```

The compared records include prompts, actions, Belief states, health, recovery,
events, and results. Reference comparison is useful for checking behavior-preserving
changes, while method tests check the mathematical and control-flow requirements.

Validation used macOS with Python 3.11, not a fresh Linux host. No paid model
experiments were run. The ClinDiag PoS template has not been tested with a real
model. ALFWorld's local TextWorld source build failed on an upstream Inform7
download; the matching Linux x86_64 wheel was downloadable, but Linux runtime
verification is still required. Full dataset preparation and MCP end-to-end
operation must be checked in the user's environment.

Keep generated outputs and workspaces out of the source package. The original
PoS MIT license does not replace the licenses of included benchmark code or data.
