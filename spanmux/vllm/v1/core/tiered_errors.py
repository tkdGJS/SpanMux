# SPDX-License-Identifier: Apache-2.0
"""Recoverable and fatal errors for the experimental tiered KV backend."""


class TieredBackendEligibilityError(RuntimeError):
    """The selected tiered path cannot execute a scheduled request.

    The engine catches only this expected control-plane failure. CUDA errors
    and other unexpected runtime failures continue to use the fatal upstream
    path because their execution state may be unsafe to recover.
    """


class TieredAllocatorInvariantError(RuntimeError):
    """A committed tiered allocator transaction violated its invariant.

    Unlike :class:`TieredBackendEligibilityError`, this error is deliberately
    not request-scoped.  Once an allocator transaction has mutated ownership,
    retrying the request without rollback could continue from a state that no
    longer matches the validated plan.  The exception must therefore escape
    the request-local recovery path and terminate the affected EngineCore.
    """
