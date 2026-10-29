"""Same-contract historical belief changes, with no outcomes or fitted weights.

The audited level adapter selects today's contract once. This helper locks that
identity for the past lookup instead of selecting another past event/threshold.
"""

import copy
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .. import sessions
from . import historical_event_features as levels

CHANGE_VERSION = "historical-event-changes-v1"
LAG_SESSIONS = 5
NY = ZoneInfo("America/New_York")
MEASUREMENTS = {
    "cpi_headline_mom": "inflation_cpi",
    "real_gdp_annualized_qoq": "jobs_growth",
    "payroll_jobs": "jobs_growth",
    "unemployment_u3": "jobs_growth",
}
IDENTITY_FIELDS = (
    "event_id", "market_id", "question", "condition_id", "yes_token_id", "no_token_id",
    "source_file", "source_sha256",
)


def binary_entropy_bits(probability):
    """Uncertainty of one exact binary question, not an equity direction."""
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("Binary entropy requires a finite probability in [0, 1].")
    if probability in (0, 1):
        return 0.0
    return -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)


def reference_origin(origin):
    """Five prior US sessions at the same conservative completed-close clock."""
    if not isinstance(origin, datetime) or origin.tzinfo is None or origin.utcoffset() is None:
        raise ValueError("Change lookup requires an aware completed-close origin.")
    local = origin.astimezone(NY)
    if local.time() != time(16, 15) or not sessions.is_us_equity_session(local.date()):
        raise ValueError("Change origin must be a US equity session at 16:15 New York time.")
    day, remaining = local.date(), LAG_SESSIONS
    while remaining:
        day -= timedelta(days=1)
        if sessions.is_us_equity_session(day):
            remaining -= 1
    return datetime.combine(day, time(16, 15), NY).astimezone(timezone.utc)


def _digest(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


class HistoricalEventChanges:
    """Network-free changes over the existing content-verified archive adapter.

    Private level primitives are intentionally reused so source, terminal, pair,
    age and complete-partition checks stay identical to the frozen level policy.
    The separately hashed source fingerprint makes that dependency explicit.
    """

    def __init__(self, adapter):
        if not isinstance(adapter, levels.HistoricalEventFeatures) or not hasattr(
            adapter, "manifest_sha256"
        ):
            raise ValueError("A loaded content-verified historical event adapter is required.")
        self.adapter = adapter
        self._policy = {
            "version": CHANGE_VERSION,
            "source_manifest_sha256": adapter.manifest_sha256,
            "level_feature_version": levels.FEATURE_VERSION,
            "lag_us_equity_sessions": LAG_SESSIONS,
            "origin_clock": "16:15 America/New_York; conservative completed regular session",
            "reference_selection": "lock current exact group, market, question and Yes/No tokens",
            "reference_upcoming_window": (
                "not reselected; current contract may then be >21 days away"
            ),
            "quote_policy": "existing available_at/cutoff/terminal checks; observed age <=24 hours",
            "creation_policy": "all locked contracts created/started by each snapshot origin",
            "fed_policy": (
                "same complete disjoint bins; common bucket per snapshot; no normalization"
            ),
            "entropy": "binary bits -p*log2(p)-(1-p)*log2(1-p); endpoints 0",
            "entropy_meaning": (
                "uncertainty of each exact binary question, never stock-up probability"
            ),
            "missing_change": "null, never asserted zero; retain available current entropy",
            "source_sha256": {
                "changes": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "levels": hashlib.sha256(Path(levels.__file__).read_bytes()).hexdigest(),
                "sessions": hashlib.sha256(Path(sessions.__file__).read_bytes()).hexdigest(),
            },
            "historical_pit_verified": False,
        }
        self.policy_sha256 = _digest(self._policy)

    @property
    def policy(self):
        return copy.deepcopy(self._policy)

    def _resolve(self, group, ref):
        key = (group, ref.get("event_id"), ref.get("market_id"))
        market = self.adapter._markets.get(key)
        if market is None or market["row"].get("quarantine_reasons"):
            raise ValueError("Current reference is not a verified eligible member of its group.")
        return market

    def _snapshot(self, market, origin, at=None):
        if market["valid_from"] > origin:
            return None, "locked_contract_not_created_at_reference_origin"
        pair, reason = self.adapter._quote_pair(market, origin, at)
        if pair is None:
            return None, reason
        return {
            "origin": origin.isoformat(),
            "yes_probability": pair[0]["probability"],
            "no_probability": pair[1]["probability"],
            "binary_entropy_bits": binary_entropy_bits(pair[0]["probability"]),
            "source_ref": self.adapter._reference(market, pair, origin),
        }, None

    def _current_snapshot(self, market, ref, origin):
        at = levels._utc(ref.get("quote_available_at"))
        if at is None:
            raise ValueError("Current reference lacks a valid availability clock.")
        snapshot, reason = self._snapshot(market, origin, at)
        if snapshot is None or any(
            snapshot["source_ref"].get(key) != ref.get(key) for key in IDENTITY_FIELDS
        ):
            raise ValueError(f"Current reference identity/quote failed verification: {reason}")
        return snapshot

    def _binary_change(self, group, market, current, past_origin):
        previous, reason = self._snapshot(market, past_origin)
        return {
            "status": "available" if previous else "missing",
            "reason_codes": [] if previous else [reason],
            "group_id": group,
            "identity": {key: current["source_ref"][key] for key in IDENTITY_FIELDS},
            "specification": asdict(market["spec"]),
            "current": current,
            "reference": previous,
            "delta_yes_probability": (
                current["yes_probability"] - previous["yes_probability"] if previous else None
            ),
        }

    def _measurement(self, channel, family, origin, past_origin):
        current = channel.get("measurements", {}).get(family, {})
        if current.get("status") != "available":
            return {
                "status": "missing", "reason_codes": ["current_measurement_unavailable"],
                "current_reason_codes": current.get("reason_codes", []),
                "current": None, "reference": None, "delta_yes_probability": None,
            }
        refs, group = current.get("source_refs", []), channel.get("group_id")
        if len(refs) != 1:
            raise ValueError("Available measurement must identify one exact binary contract.")
        market = self._resolve(group, refs[0])
        if market["spec"] is None or market["spec"].family != family or (
            asdict(market["spec"]) != current.get("specification")
        ):
            raise ValueError("Current selected measurement specification changed.")
        snapshot = self._current_snapshot(market, refs[0], origin)
        if any(snapshot[key] != current.get(key) for key in ("yes_probability", "no_probability")):
            raise ValueError("Current selected measurement probability changed.")
        return self._binary_change(group, market, snapshot, past_origin)

    def _fed(self, current, origin, past_origin):
        missing = {
            "status": "missing", "reason_codes": ["current_fed_partition_unavailable"],
            "current": None, "reference": None, "delta_probabilities": None, "components": [],
        }
        if current.get("status") != "available":
            return {**missing, "current_reason_codes": current.get("reason_codes", [])}
        group, refs = current.get("group_id"), current.get("source_refs", [])
        markets = [self._resolve(group, ref) for ref in refs]
        canonical = self.adapter._canonical(self.adapter._group_markets.get(group, []))
        if not markets or len(markets) != len(canonical) or any(
            left is not right for left, right in zip(markets, canonical)
        ):
            raise ValueError("Current Fed references do not cover its complete canonical bins.")
        verified = self.adapter._fed(markets, origin)
        if verified != {key: value for key, value in current.items() if key not in (
            "group_id", "declared_event_end", "days_to_declared_end", "end_time_verified_as_release"
        )}:
            raise ValueError("Current Fed partition differs from its verified source.")
        # Fixed identities only: _fed does not choose a different release group.
        previous = self.adapter._fed(markets, past_origin)
        available = previous.get("status") == "available"
        components = []
        for market, ref in zip(markets, refs):
            snapshot = self._current_snapshot(market, ref, origin)
            change = self._binary_change(group, market, snapshot, past_origin)
            if available:
                # Each component must use the aggregate's common reference bucket,
                # rather than its individually latest (possibly different) bucket.
                previous_ref = previous["source_refs"][len(components)]
                at = levels._utc(previous_ref["quote_available_at"])
                prior, _ = self._snapshot(market, past_origin, at)
                change["status"], change["reason_codes"] = "available", []
                change["reference"] = prior
                change["delta_yes_probability"] = snapshot["yes_probability"] - prior[
                    "yes_probability"
                ]
            else:
                change["status"] = "missing"
                change["reason_codes"] = ["reference_fed_partition_unavailable"]
                change["delta_yes_probability"] = None
            components.append(change)
        return {
            "status": "available" if available else "missing",
            "reason_codes": [] if available else previous.get("reason_codes", []),
            "group_id": group,
            "current": {"origin": origin.isoformat(), **verified},
            "reference": {"origin": past_origin.isoformat(), **previous},
            "delta_probabilities": {
                key: verified["probabilities"][key] - previous["probabilities"][key]
                for key in ("cut", "hold", "hike")
            } if available else None,
            "components": components,
        }

    def lookup(self, origin):
        past_origin = reference_origin(origin)
        origin = origin.astimezone(timezone.utc)
        current = self.adapter.lookup(origin)  # Exactly one semantic contract selection.
        if current.get("source_manifest_sha256") != self._policy["source_manifest_sha256"]:
            raise ValueError("Current lookup changed the bound source archive.")
        channels = current["channels"]
        return copy.deepcopy({
            "feature_version": CHANGE_VERSION,
            "origin": origin.isoformat(),
            "reference_origin": past_origin.isoformat(),
            "source_policy_sha256": self.policy_sha256,
            "source_manifest_sha256": self._policy["source_manifest_sha256"],
            "fed": self._fed(channels.get("fed", {}), origin, past_origin),
            "measurements": {
                family: self._measurement(channels.get(channel, {}), family, origin, past_origin)
                for family, channel in MEASUREMENTS.items()
            },
            "current_snapshot": current,
            "historical_pit_verified": False,
            "limitations": [
                "Current metadata/rules are not authenticated historical publications.",
                "Reference quotes retain the level adapter's bucket-end availability proxy.",
                "Missing changes remain null; entropy measures exact-question uncertainty only.",
                "No stock direction, outcome, fitted weight or economic inverse of No is inferred.",
            ],
        })
