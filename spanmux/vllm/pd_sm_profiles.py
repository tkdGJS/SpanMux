# SPDX-License-Identifier: Apache-2.0
"""Pure helpers shared by the dynamic P/D SM controller and GPU worker.

This module intentionally has no torch or vLLM imports.  Engine-side policy
tests can therefore validate profile parsing and selection on CPU-only hosts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class PDSMProfile:
    """One immutable decode/prefill SM partition."""

    decode_sms: int
    prefill_sms: int

    def __post_init__(self) -> None:
        if self.decode_sms < 0 or self.prefill_sms < 0:
            raise ValueError("P/D SM counts must be non-negative")
        if self.decode_sms == 0 and self.prefill_sms == 0:
            raise ValueError("at least one P/D SM count must be positive")

    @property
    def profile_id(self) -> str:
        return f"d{self.decode_sms}_p{self.prefill_sms}"

    @property
    def total_sms(self) -> int:
        return self.decode_sms + self.prefill_sms

    @property
    def is_decode_only(self) -> bool:
        return self.decode_sms > 0 and self.prefill_sms == 0

    @property
    def is_prefill_only(self) -> bool:
        return self.decode_sms == 0 and self.prefill_sms > 0

    @property
    def is_concurrent(self) -> bool:
        return self.decode_sms > 0 and self.prefill_sms > 0


def _parse_profile_item(item: str) -> PDSMProfile:
    normalized = item.strip().lower()
    if not normalized:
        raise ValueError("empty P/D SM profile")
    if "/" in normalized:
        decode_text, prefill_text = normalized.split("/", 1)
    elif normalized.startswith("d") and "_p" in normalized:
        decode_text, prefill_text = normalized[1:].split("_p", 1)
    else:
        raise ValueError(
            f"invalid P/D SM profile {item!r}; expected DECODE/PREFILL"
        )
    try:
        return PDSMProfile(int(decode_text), int(prefill_text))
    except ValueError as exc:
        raise ValueError(f"invalid P/D SM profile {item!r}: {exc}") from exc


def auto_pd_sm_profiles(total_sms: int, alignment: int = 2) -> tuple[PDSMProfile, ...]:
    """Build coarse solo and concurrent profiles for one device.

    Fractions are chosen so a 40-SM T4 produces exactly:
    0/40, 8/32, 16/24, 20/20, 24/16, 32/8, 40/0.
    """

    if total_sms <= 0:
        raise ValueError("total_sms must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if total_sms % alignment != 0:
        raise ValueError(
            f"total_sms={total_sms} is not aligned to {alignment} SMs"
        )

    decode_counts = {0, total_sms}
    for numerator in (1, 2, 3, 4):
        raw = total_sms * numerator / 5
        aligned = int(round(raw / alignment)) * alignment
        if 0 < aligned < total_sms:
            decode_counts.add(aligned)
    half = int(round((total_sms / 2) / alignment)) * alignment
    if 0 < half < total_sms:
        decode_counts.add(half)

    return tuple(
        PDSMProfile(decode_sms, total_sms - decode_sms)
        for decode_sms in sorted(decode_counts)
    )


def dense_pd_sm_profiles(
    total_sms: int, alignment: int = 2
) -> tuple[PDSMProfile, ...]:
    """Build every full-device partition at ``alignment`` granularity.

    A 40-SM T4 with alignment two therefore produces 0/40, 2/38, ...,
    38/2, and 40/0.  This is opt-in because creating every GreenContext
    partition has a larger startup and bookkeeping cost than the legacy
    coarse ``auto`` set.
    """

    if total_sms <= 0:
        raise ValueError("total_sms must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if total_sms % alignment != 0:
        raise ValueError(
            f"total_sms={total_sms} is not aligned to {alignment} SMs"
        )
    return tuple(
        PDSMProfile(decode_sms, total_sms - decode_sms)
        for decode_sms in range(0, total_sms + 1, alignment)
    )


def parse_pd_sm_profiles(
    spec: str,
    *,
    total_sms: int,
    alignment: int = 2,
    require_full_device: bool = True,
) -> tuple[PDSMProfile, ...]:
    """Parse a comma-separated profile list, or generate the ``auto`` list."""

    normalized = spec.strip().lower()
    if not normalized or normalized == "auto":
        profiles = auto_pd_sm_profiles(total_sms, alignment)
    elif normalized in {"auto_dense", "dense"}:
        profiles = dense_pd_sm_profiles(total_sms, alignment)
    else:
        profiles = tuple(_parse_profile_item(item) for item in spec.split(","))

    unique: dict[str, PDSMProfile] = {}
    for profile in profiles:
        if profile.total_sms > total_sms:
            raise ValueError(
                f"profile {profile.profile_id} exceeds total_sms={total_sms}"
            )
        if require_full_device and profile.total_sms != total_sms:
            raise ValueError(
                f"profile {profile.profile_id} uses {profile.total_sms} SMs; "
                f"expected all {total_sms}"
            )
        for count in (profile.decode_sms, profile.prefill_sms):
            if count and count % alignment != 0:
                raise ValueError(
                    f"profile {profile.profile_id} is not aligned to "
                    f"{alignment} SMs"
                )
        unique[profile.profile_id] = profile

    result = tuple(sorted(unique.values()))
    if not any(profile.is_decode_only for profile in result):
        raise ValueError("profile list must include a decode-only split")
    if not any(profile.is_prefill_only for profile in result):
        raise ValueError("profile list must include a prefill-only split")
    if not any(profile.is_concurrent for profile in result):
        raise ValueError("profile list must include a concurrent P/D split")
    return result


def profile_by_id(
    profiles: tuple[PDSMProfile, ...], profile_id: str
) -> PDSMProfile | None:
    return next(
        (profile for profile in profiles if profile.profile_id == profile_id),
        None,
    )


def closest_profile(
    profiles: tuple[PDSMProfile, ...],
    *,
    decode_sms: int,
    prefill_sms: int,
    concurrent_only: bool = False,
) -> PDSMProfile:
    candidates = tuple(
        profile
        for profile in profiles
        if not concurrent_only or profile.is_concurrent
    )
    if not candidates:
        raise ValueError("no eligible P/D SM profile")
    return min(
        candidates,
        key=lambda profile: (
            abs(profile.decode_sms - decode_sms)
            + abs(profile.prefill_sms - prefill_sms),
            abs(profile.decode_sms - profile.prefill_sms),
            profile.profile_id,
        ),
    )
