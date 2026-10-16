# ADR 0003: Download does not imply model executability

Status: Accepted for v0.6.

All public model artifacts pass through `ModelCompatibilityInspector`. Only recognized formats with an existing safe adapter become `READY`; opaque checkpoints remain `ADAPTER_REQUIRED`, `AMBIGUOUS`, or `UNSUPPORTED`. Registration retains the artifact even when execution is unavailable.
