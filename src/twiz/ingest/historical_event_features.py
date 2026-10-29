"""Typed, origin-bounded event features from a content-verified research archive.

These are reconstructed token beliefs for exact economic questions. They are not
stock probabilities, authenticated historical publications, or fitted weights.
"""

import bisect
import copy
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

FEATURE_VERSION = "historical-event-features-v1"
MAX_FILES, MAX_BYTES, MAX_CONTRACTS, MAX_POINTS = 2500, 256 * 1024 * 1024, 1000, 1000000
FAMILIES = {
    "fed": ("policy_rate_change",),
    "inflation_cpi": ("cpi_headline_mom", "cpi_headline_yoy", "cpi_core_mom", "cpi_core_yoy"),
    "jobs_growth": ("real_gdp_annualized_qoq", "payroll_jobs", "unemployment_u3"),
}
BENCHMARKS = {
    "cpi_headline_mom": 0.2,
    "cpi_core_mom": 0.2,
    "cpi_headline_yoy": 3.0,
    "cpi_core_yoy": 3.0,
    "real_gdp_annualized_qoq": 0.0,
    "payroll_jobs": 150000.0,
    "unemployment_u3": 4.0,
}


def _utc(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _array(value):
    result = json.loads(value) if isinstance(value, str) else value
    return result if isinstance(result, list) else []


@dataclass(frozen=True)
class QuestionSpec:
    family: str
    unit: str
    relation: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    lower_inclusive: Optional[bool] = None
    upper_inclusive: Optional[bool] = None
    rate_direction: Optional[str] = None
    basis: str = ""

    @property
    def key(self):
        return (
            self.family,
            self.unit,
            self.relation,
            self.lower,
            self.upper,
            self.lower_inclusive,
            self.upper_inclusive,
            self.rate_direction,
            self.basis,
        )


def parse_question(question: str, description: str) -> tuple:
    """Explicit source grammar only. Unsupported typos/conditional fallbacks stay unknown."""
    if not isinstance(question, str) or not isinstance(description, str) or not description.strip():
        return None, "missing_question_or_rules"
    q, rules = question.lower().strip(), description.lower()
    negation_text = re.sub(r"\bno change\b", "", q)
    if re.search(r"\bno\b|\bnot\b|\bnever\b", negation_text):
        return None, "negated_question_not_reinterpreted"
    compound = re.sub(r"\bor (?:more|less|higher|lower)\b", "", q)
    compound = re.sub(r"between [-\d.,k% ]+\s+and\s+[-\d.,k% ]+", "", compound)
    if re.search(r"\band\b|\bor\b", compound):
        return None, "compound_question_not_reinterpreted"
    if (
        re.search(r"\bif\b|\bunless\b|\bdissent\b|\bcombo\b|\banother level\b", q)
        or "last available month" in rules
        or "last available month's" in rules
    ):
        return None, "conditional_or_other_level_contract"
    if re.search(r"\bfed\b|federal reserve", q) and re.search(r"rates?", q):
        if "upper bound" not in rules or "federal funds" not in rules:
            return None, "fed_rate_basis_unverified"
        if "%" in q or len(re.findall(r"-?\d+(?:\.\d+)?\+?\s*bps\b", q)) > 1:
            return None, "incompatible_unit_or_multiple_rate_thresholds"
        if re.search(r"\bno change\b|\bunchanged\b", q):
            if re.search(r"\d+(?:\.\d+)?\s*bps", q):
                return None, "contradictory_no_change_magnitude"
            return QuestionSpec(
                "policy_rate_change",
                "basis_points",
                "eq",
                0,
                0,
                True,
                True,
                "hold",
                "target_federal_funds_upper_bound",
            ), None
        change = re.search(
            r"\b(decrease|decreases|cut|increase|increases|hike)\b.*?"
            r"\b(\d+(?:\.\d+)?)(\+)?\s*bps\b",
            q,
        )
        if not change:
            return None, "unsupported_fed_bin"
        direction = "cut" if change[1] in ("decrease", "decreases", "cut") else "hike"
        amount = float(change[2])
        if amount <= 0 or re.search(r"-\d+(?:\.\d+)?\s*bps", q):
            return None, "invalid_rate_change_magnitude"
        if direction == "cut":
            return QuestionSpec(
                "policy_rate_change",
                "basis_points",
                "le" if change[3] else "eq",
                None if change[3] else -amount,
                -amount,
                None if change[3] else True,
                True,
                direction,
                "target_federal_funds_upper_bound",
            ), None
        return QuestionSpec(
            "policy_rate_change",
            "basis_points",
            "ge" if change[3] else "eq",
            amount,
            None if change[3] else amount,
            True,
            None if change[3] else True,
            direction,
            "target_federal_funds_upper_bound",
        ), None
    if "unemployment" in q:
        if "u-3" not in rules:
            return None, "unemployment_measure_unverified"
        family, unit, basis = "unemployment_u3", "percent", "seasonally_adjusted_U3_first_release"
    elif re.search(r"\bgdp\b", q):
        if "real gdp" not in rules and "annualized gdp" not in rules:
            return None, "gdp_measure_unverified"
        if "annualized" not in rules and "annual rate" not in rules:
            return None, "gdp_annualization_unverified"
        family, unit, basis = (
            "real_gdp_annualized_qoq",
            "percent",
            "real_GDP_annualized_qoq_advance",
        )
        if re.search(r"\bgdp declines\b", q) and not re.search(r"%|\d\s*k\b", q):
            return QuestionSpec(family, unit, "lt", None, 0.0, None, False, basis=basis), None
    elif "jobs" in q:
        if "%" in q:
            return None, "incompatible_question_unit"
        if "nonfarm payroll" not in rules:
            return None, "jobs_measure_unverified"
        family, unit, basis = "payroll_jobs", "jobs", "nonfarm_payroll_first_release"
        if re.search(r"\blose jobs\b", q):
            if re.search(r"\d\s*k\b", q):
                return None, "unsupported_job_loss_magnitude"
            return QuestionSpec(family, unit, "lt", None, 0.0, None, False, basis=basis), None
    elif "inflation" in q or "cpi" in q:
        if "consumer price" not in rules and "cpi-u" not in rules:
            return None, "cpi_measure_unverified"
        core = "core cpi" in q or "excluding food and energy" in rules
        yoy = bool(re.search(r"\byoy\b|year.over.year|12.month|twelve.month", q + " " + rules))
        mom = bool(
            re.search(
                r"\bmom\b|month.to.month|one.month|month.over.month|from .+ to .+20\d\d",
                q + " " + rules,
            )
        )
        if yoy == mom:
            return None, "cpi_frequency_ambiguous"
        family = "cpi_" + ("core_" if core else "headline_") + ("yoy" if yoy else "mom")
        unit, basis = (
            "percent",
            "CPI_U_" + ("core" if core else "headline") + ("_yoy" if yoy else "_mom"),
        )
    else:
        return None, "unsupported_question_family"

    if (family == "payroll_jobs" and "%" in q) or (
        family != "payroll_jobs" and re.search(r"\d\s*k\b", q)
    ):
        return None, "incompatible_question_unit"
    comparisons = re.findall(r"less than|more than|greater than|below|above|at least|[<>]", q)
    if len(comparisons) + bool(re.search(r"\bbetween\b", q)) > 1:
        return None, "multiple_comparison_clauses"

    numeric = r"-?\d[\d,]*(?:\.\d+)?"
    between = re.search(
        r"(?:between\s+|(?:add|gain)\s+)(" + numeric + r")\s*(k|%)?\s*"
        r"(?:and|to|-)\s*(" + numeric + r")\s*(k|%)?",
        q,
    )
    if between:
        a, b = float(between[1].replace(",", "")), float(between[3].replace(",", ""))
        if family == "payroll_jobs" and "k" in (between[2], between[4]):
            a, b = a * 1000, b * 1000
        lower, upper = min(a, b), max(a, b)
        if lower == upper:
            return None, "empty_interval"
        # The question's displayed endpoints remain literal: never turn 199k into 200k.
        explicit = {}
        for match in re.finditer(r"(" + numeric + r")\s*(k|%)?\s*\((inclusive|exclusive)\)", rules):
            value = float(match[1].replace(",", ""))
            if family == "payroll_jobs" and match[2] == "k":
                value *= 1000
            explicit[value] = match[3]
        inclusive = (
            (True, False)
            if explicit.get(lower) == "inclusive" and explicit.get(upper) == "exclusive"
            else (None, None)
        )
        if inclusive == (None, None):
            return None, "interval_endpoint_inclusion_unverified"
        return QuestionSpec(family, unit, "interval", lower, upper, *inclusive, basis=basis), None
    comparison = re.search(
        r"(less than|more than|greater than|below|above|at least|[<>])\s*("
        + numeric
        + r")\s*(k|%)?",
        q,
    )
    if comparison:
        operator = {
            "less than": "lt",
            "below": "lt",
            "<": "lt",
            "more than": "gt",
            "greater than": "gt",
            "above": "gt",
            ">": "gt",
            "at least": "ge",
        }[comparison[1]]
        value = float(comparison[2].replace(",", ""))
        if family == "payroll_jobs" and comparison[3] == "k":
            value *= 1000
    else:
        tail = re.search(r"(" + numeric + r")\s*(k|%)\s*(or more|or higher|or lower|or less)?", q)
        if not tail:
            return None, "unsupported_or_malformed_comparison"
        value = float(tail[1].replace(",", ""))
        if family == "payroll_jobs" and tail[2] == "k":
            value *= 1000
        operator = {"or more": "ge", "or higher": "ge", "or lower": "le", "or less": "le"}.get(
            tail[3]
        )
        if operator is None:
            if family != "unemployment_u3":
                return None, "comparison_relation_unverified"
            operator = "eq"
    if operator in ("lt", "le"):
        return QuestionSpec(
            family, unit, operator, None, value, None, operator == "le", basis=basis
        ), None
    if operator in ("gt", "ge"):
        return QuestionSpec(
            family, unit, operator, value, None, operator == "ge", None, basis=basis
        ), None
    return QuestionSpec(family, unit, "eq", value, value, True, True, basis=basis), None


class HistoricalEventFeatures:
    """Load one sealed dataset once; each lookup is bounded and network-free."""

    @classmethod
    def from_archive(cls, path: Path, expected_manifest_sha256: str):
        try:
            return cls._from_archive(path, expected_manifest_sha256)
        except (KeyError, TypeError, IndexError, OverflowError, OSError) as exc:
            raise ValueError("Malformed or inaccessible historical event archive.") from exc

    @classmethod
    def _from_archive(cls, path: Path, expected_manifest_sha256: str):
        self = cls()
        root = Path(path).resolve()
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest_sha256)):
            raise ValueError("An externally recorded manifest SHA256 is required.")
        body = (root / "manifest.json").read_bytes()
        if hashlib.sha256(body).hexdigest() != expected_manifest_sha256:
            raise ValueError("Archive manifest does not match the expected hash.")
        manifest = json.loads(body)
        names = manifest.get("files")
        if not isinstance(names, dict) or len(names) > MAX_FILES:
            raise ValueError("Invalid or oversized archive file inventory.")
        self._files, self._digests, total = {}, {}, 0
        for name, digest in names.items():
            target = root / name
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                raise ValueError("Unsafe archive file reference.")
            total += target.stat().st_size
            if total > MAX_BYTES:
                raise ValueError("Archive byte budget exceeded.")
            content = target.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Archive file hash mismatch: {name}")
            self._files[name] = content
            self._digests[name] = digest
        self.manifest_sha256, self._cache = expected_manifest_sha256, {}
        self._analysis_clock = _utc(manifest.get("analysis_clock_utc"))
        if self._analysis_clock is None:
            raise ValueError("Archive analysis clock is missing its timezone.")
        index = self._json("index.json")
        rows = index.get("contracts")
        if not isinstance(rows, list) or len(rows) > MAX_CONTRACTS:
            raise ValueError("Invalid or oversized contract inventory.")
        groups = index.get("groups")
        if not isinstance(groups, list) or len(groups) > 100:
            raise ValueError("Invalid release group inventory.")
        self._plans = {group["group_id"]: group for group in groups}
        self._markets, self._group_markets, tokens, count = {}, {}, set(), 0
        for row in rows:
            token = str(row.get("token_id"))
            if token in tokens:
                raise ValueError("Duplicate outcome token in archive index.")
            tokens.add(token)
            source, parent = self._validate_source(row)
            group_id = row["release_group_id"]
            market_key = (group_id, row["event_id"], row["market_id"])
            if market_key not in self._markets:
                spec, reason = parse_question(source.get("question"), source.get("description"))
                if spec and spec.family not in FAMILIES.get(row["channel"], ()):
                    spec, reason = None, "family_channel_mismatch"
                self._markets[market_key] = {
                    "row": row,
                    "source": source,
                    "parent": parent,
                    "spec": spec,
                    "parse_reason": reason,
                    "quotes": {},
                    "valid_from": _utc(row["valid_from"]),
                    "cutoff": _utc(row["cutoff"]),
                }
                self._group_markets.setdefault(group_id, []).append(self._markets[market_key])
            market = self._markets[market_key]
            quotes = self._validate_quotes(row, source)
            count += len(quotes)
            if count > MAX_POINTS:
                raise ValueError("Historical quote count budget exceeded.")
            if row["outcome"] in market["quotes"]:
                raise ValueError("Duplicate market/outcome membership.")
            market["quotes"][row["outcome"]] = quotes
        for group_id, plan in self._plans.items():
            markets = self._group_markets.get(group_id, [])
            if {m["row"]["event_id"] for m in markets} != set(plan["parent_event_ids"]):
                raise ValueError("Incomplete release group event inventory.")
            for market in markets:
                parent = market["parent"]
                expected = {str(m["id"]) for m in parent["markets"]}
                actual = {
                    m["row"]["market_id"]
                    for m in markets
                    if m["row"]["event_id"] == str(parent["id"])
                }
                if actual != expected or set(market["quotes"]) != {"Yes", "No"}:
                    raise ValueError("Incomplete event market/outcome inventory.")
        return self

    def _json(self, name):
        if name not in self._files:
            raise ValueError(f"Unsealed archive reference: {name}")
        if name not in self._cache:
            self._cache[name] = json.loads(self._files[name])
        return self._cache[name]

    def _validate_source(self, row):
        wrapper = self._json(row["source_file"])
        if hashlib.sha256(self._files[row["source_file"]]).hexdigest() != row["source_sha256"]:
            raise ValueError("Projected source hash mismatch.")
        proof = wrapper.get("canonical_source") or {}
        discovery_manifest = self._json("inputs/discovery/manifest.json")
        if hashlib.sha256(self._files["inputs/discovery/manifest.json"]).hexdigest() != proof.get(
            "manifest_sha256"
        ):
            raise ValueError("Canonical discovery manifest binding mismatch.")
        raw = self._json(proof["body_file"])
        if hashlib.sha256(self._files[proof["body_file"]]).hexdigest() != proof["body_sha256"]:
            raise ValueError("Canonical search body hash mismatch.")
        receipt = self._json(proof["receipt_file"])
        url = urlparse(str(receipt.get("request_url", "")))
        if (
            url.scheme != "https"
            or url.netloc != "gamma-api.polymarket.com"
            or url.path != "/public-search"
            or wrapper.get("request_url") != receipt["request_url"]
            or wrapper.get("fetched_at") != receipt.get("fetched_at")
            or receipt.get("body_sha256") != proof["body_sha256"]
            or "inputs/discovery/" + receipt.get("body_file", "") != proof["body_file"]
            or discovery_manifest.get("files", {}).get(
                proof["body_file"].removeprefix("inputs/discovery/")
            )
            != proof["body_sha256"]
        ):
            raise ValueError("Canonical public-search receipt mismatch.")
        parent = raw["events"][proof["event_array_index"]]
        if wrapper.get("response") != [parent] or str(parent["id"]) != row["event_id"]:
            raise ValueError("Source event membership mismatch.")
        plan = self._plans.get(row.get("release_group_id"))
        if (
            not plan
            or row["event_id"] not in plan.get("parent_event_ids", [])
            or row.get("channel") != plan.get("channel")
        ):
            raise ValueError("Release group membership mismatch.")
        found = [
            market
            for market in parent.get("markets", [])
            if str(market.get("id")) == row["market_id"]
        ]
        if len(found) != 1:
            raise ValueError("Source market membership mismatch.")
        market = found[0]
        labels, ids = _array(market.get("outcomes")), _array(market.get("clobTokenIds"))
        if (
            set(labels) != {"Yes", "No"}
            or len(labels) != 2
            or len(ids) != 2
            or len(set(str(token) for token in ids)) != 2
            or row.get("outcome") not in labels
            or str(ids[labels.index(row["outcome"])]) != row.get("token_id")
            or row.get("question") != market.get("question")
            or row.get("contract_metadata", {}).get("condition_id") != market.get("conditionId")
        ):
            raise ValueError("Source question/outcome/token binding mismatch.")
        starts = [
            _utc(item.get(key))
            for item in (parent, market)
            for key in ("createdAt", "creationDate", "startDate", "published_at")
        ]
        if any(
            item.get(key) is not None and _utc(item[key]) is None
            for item in (parent, market)
            for key in ("createdAt", "creationDate", "startDate", "published_at")
        ):
            raise ValueError("Naive or invalid source creation/start timestamp.")
        starts = [value for value in starts if value is not None]
        ends = [_utc(item.get("endDate")) for item in (parent, market)]
        closures = [_utc(market.get(key)) for key in ("closedTime", "umaEndDate", "resolvedAt")]
        if any(
            market.get(key) is not None and _utc(market[key]) is None
            for key in ("closedTime", "umaEndDate", "resolvedAt")
        ):
            raise ValueError("Naive or invalid source closure timestamp.")
        closures = [value for value in closures if value is not None]
        if (
            not starts
            or any(value is None for value in ends)
            or _utc(market.get("createdAt")) is None
            or _utc(row.get("valid_from")) != max(starts)
            or _utc(row.get("source_end_date")) != min(ends)
            or _utc(row.get("cutoff")) != min(ends + closures + [self._analysis_clock])
        ):
            raise ValueError("Source creation/start/end boundary mismatch.")
        return market, parent

    def _validate_quotes(self, row, source):
        if row.get("status") not in ("downloaded", "reused_verified"):
            return {}
        record = self._json(row["history_file"])
        if (
            record.get("token_id") != row["token_id"]
            or record.get("market_id") != row["market_id"]
            or record.get("outcome") != row["outcome"]
            or record.get("contract_metadata", {}).get("condition_id") != source.get("conditionId")
        ):
            raise ValueError("History identity mismatch.")
        start, cutoff = _utc(row["valid_from"]), _utc(row["cutoff"])
        points = record.get("history")
        if not isinstance(points, list) or len(points) > 30000:
            raise ValueError("Invalid or oversized token history.")
        responses = record.get("response_files")
        if (
            not isinstance(responses, list)
            or not 1 <= len(responses) <= 3
            or len(set(responses)) != len(responses)
            or record.get("pagination_complete") is not True
        ):
            raise ValueError("Missing or incomplete owned response-file inventory.")
        quotes = {}
        terminal = None
        owned, next_cursor = set(), None
        for page, raw_name in enumerate(responses):
            raw_path = str(Path(row.get("raw_path_base", "")) / raw_name)
            envelope = self._json(raw_path)
            receipt = self._json(raw_path.removesuffix(".body") + ".json")
            url = urlparse(receipt["request_url"])
            query = parse_qs(url.query)
            expected_cursor = [next_cursor] if next_cursor is not None else None
            if (
                url.scheme != "https"
                or url.netloc != "data-api.polymarket.com"
                or url.path != "/v2/prices-history"
                or query.get("token_id") != [row["token_id"]]
                or receipt.get("query", {}).get("token_id") != row["token_id"]
                or query.get("cursor") != expected_cursor
                or _utc(receipt.get("fetched_at")) is None
            ):
                raise ValueError("Owned raw token-history receipt mismatch.")
            data, pagination = envelope.get("data"), envelope.get("pagination")
            if not isinstance(data, list) or len(data) > 10000 or not isinstance(pagination, dict):
                raise ValueError("Invalid owned history response envelope.")
            next_cursor = pagination.get("next_cursor")
            last = page == len(responses) - 1
            if (last and (pagination.get("has_more") is not False or next_cursor is not None)) or (
                not last
                and (
                    pagination.get("has_more") is not True
                    or not isinstance(next_cursor, str)
                    or not next_cursor
                )
            ):
                raise ValueError("Incomplete owned response pagination.")
            owned.add(raw_path)
            for raw in data:
                if not isinstance(raw, dict) or _number(raw.get("resolution_seconds")) != 0:
                    continue
                tick, price = _number(raw.get("timestamp")), _number(raw.get("price"))
                if tick is None or price is None or not 0 <= price <= 1:
                    continue
                observed = datetime.fromtimestamp(tick, timezone.utc)
                if start <= observed < cutoff:
                    terminal = observed if terminal is None else min(terminal, observed)
        for point in points:
            price, tick, resolution = (
                _number(point.get(key)) for key in ("p", "t", "resolution_seconds")
            )
            observed, available = _utc(point.get("observed_at")), _utc(point.get("available_at"))
            if (
                price is None
                or not 0 <= price <= 1
                or tick is None
                or resolution is None
                or tick != int(tick)
                or resolution <= 0
                or resolution != int(resolution)
                or observed is None
                or available is None
                or observed.timestamp() != tick
                or not start <= observed < cutoff
                or not observed + timedelta(seconds=max(10800, resolution)) <= available <= cutoff
                or not point.get("raw_references")
                or terminal is not None
                and (observed >= terminal or available > terminal)
            ):
                raise ValueError("Invalid historical quote/time boundary.")
            refs = []
            for ref in point["raw_references"]:
                body_path = str(Path(row.get("raw_path_base", "")) / ref["body_file"])
                if body_path not in owned:
                    raise ValueError("Quote reference is not in its owned response_files.")
                offset = ref["data_index"]
                if (
                    not isinstance(offset, int)
                    or isinstance(offset, bool)
                    or offset < 0
                    or offset >= len(self._json(body_path)["data"])
                ):
                    raise ValueError("Invalid raw quote data index.")
                raw = self._json(body_path)["data"][ref["data_index"]]
                receipt = self._json(body_path.removesuffix(".body") + ".json")
                url = urlparse(receipt["request_url"])
                query = parse_qs(url.query)
                if (
                    url.scheme != "https"
                    or url.netloc != "data-api.polymarket.com"
                    or url.path != "/v2/prices-history"
                    or query.get("token_id") != [row["token_id"]]
                    or receipt.get("query", {}).get("token_id") != row["token_id"]
                    or _number(raw.get("timestamp")) != tick
                    or _number(raw.get("price")) != price
                    or _number(raw.get("resolution_seconds")) != resolution
                ):
                    raise ValueError("Raw token history reference mismatch.")
                refs.append(
                    {
                        "body_file": body_path,
                        "body_sha256": self._digests[body_path],
                        "data_index": ref["data_index"],
                        "fetched_at": receipt.get("fetched_at"),
                    }
                )
            if available in quotes:
                raise ValueError("Ambiguous duplicate quote availability time.")
            quotes[available] = {
                "probability": price,
                "observed_at": observed,
                "available_at": available,
                "token_id": row["token_id"],
                "raw_references": refs,
            }
        return quotes

    def schema_inventory(self):
        supported, unsupported, supported_with_history = Counter(), Counter(), Counter()
        for market in self._markets.values():
            if market["spec"]:
                supported[market["spec"].family] += 1
                if market["row"].get("status") in ("downloaded", "reused_verified"):
                    supported_with_history[market["spec"].family] += 1
            else:
                unsupported[market["parse_reason"]] += 1
        return {
            "unique_markets": len(self._markets),
            "supported": dict(supported),
            "unsupported": dict(unsupported),
            "supported_with_downloaded_history": dict(supported_with_history),
            "feature_version": FEATURE_VERSION,
        }

    @staticmethod
    def _missing(reason):
        return {
            "status": "missing",
            "reason_codes": [reason],
            "source_refs": [],
            "historical_pit_verified": False,
        }

    @staticmethod
    def _canonical(markets):
        unique = {}
        for market in sorted(
            markets,
            key=lambda m: (m["valid_from"], int(m["row"]["event_id"]), int(m["row"]["market_id"])),
        ):
            key = (
                market["spec"].key
                if market["spec"]
                else ("unsupported", market["source"]["question"])
            )
            unique.setdefault(key, market)
        return list(unique.values())

    def _quote_pair(self, market, origin, at=None):
        if origin >= market["cutoff"]:
            return None, "market_cutoff_has_passed"
        yes, no = market["quotes"].get("Yes", {}), market["quotes"].get("No", {})
        common = sorted(set(yes) & set(no))
        if at is None:
            offset = bisect.bisect_right(common, origin)
            if not offset:
                return None, "no_available_quote_at_origin"
            at = common[offset - 1]
        if at not in yes or at not in no:
            return None, "no_common_yes_no_quote"
        pair = (yes[at], no[at])
        if (
            any(
                quote["observed_at"] > origin or origin - quote["observed_at"] > timedelta(hours=24)
                for quote in pair
            )
            or at > origin
        ):
            return None, "quote_older_than_24h_or_future"
        if abs(sum(quote["probability"] for quote in pair) - 1) > 0.02 + 1e-12:
            return None, "incoherent_yes_no_sum"
        return pair, None

    def _reference(self, market, pair, origin):
        row = market["row"]
        return {
            "event_id": row["event_id"],
            "market_id": row["market_id"],
            "question": market["source"]["question"],
            "source_file": row["source_file"],
            "source_sha256": row["source_sha256"],
            "condition_id": market["source"].get("conditionId"),
            "yes_token_id": pair[0]["token_id"],
            "no_token_id": pair[1]["token_id"],
            "quote_available_at": pair[0]["available_at"].isoformat(),
            "yes_observed_at": pair[0]["observed_at"].isoformat(),
            "no_observed_at": pair[1]["observed_at"].isoformat(),
            "age_hours": max(
                (origin - quote["observed_at"]).total_seconds() / 3600 for quote in pair
            ),
            "raw_quote_references": pair[0]["raw_references"] + pair[1]["raw_references"],
            "rules_version": "current_download_not_verified_historical",
            "historical_publication_time_verified": False,
        }

    def _measurement(self, markets, family, origin):
        candidates = self._canonical(
            [
                market
                for market in markets
                if market["spec"]
                and market["valid_from"] <= origin
                and market["spec"].family == family
            ]
        )
        if not candidates:
            return self._missing("no_supported_measurement_in_selected_group")
        benchmark = BENCHMARKS[family]

        def rank(market):
            spec = market["spec"]
            distance = min(
                abs(value - benchmark) for value in (spec.lower, spec.upper) if value is not None
            )
            return (
                round(distance, 9),
                spec.relation == "interval",
                int(market["row"]["market_id"]),
            )

        chosen = min(candidates, key=rank)
        # Selection is semantic and fixed before reading quote availability/probabilities.
        if chosen["valid_from"] > origin:
            return self._missing("reference_contract_not_created_at_origin")
        pair, reason = self._quote_pair(chosen, origin)
        result = {
            "family": family,
            "specification": asdict(chosen["spec"]),
            "benchmark": benchmark,
            "benchmark_distance": rank(chosen)[0],
            "selection_rule": "nearest_numeric_boundary_then_one_sided_then_market_id",
            "yes_probability": None,
            "no_probability": None,
            "probability_meaning": (
                "Exact quoted question; benchmark is a reference, "
                "and no stock direction is inferred."
            ),
            "historical_pit_verified": False,
        }
        if pair is None:
            return {**result, **self._missing(reason)}
        return {
            **result,
            "status": "available",
            "reason_codes": [],
            "yes_probability": pair[0]["probability"],
            "no_probability": pair[1]["probability"],
            "source_refs": [self._reference(chosen, pair, origin)],
        }

    def _fed(self, markets, origin):
        markets = self._canonical(markets)
        if any(market["spec"] is None or market["valid_from"] > origin for market in markets):
            return {
                **self._missing("unsupported_or_not_yet_created_rate_bin"),
                "status": "unsupported",
                "probabilities": None,
            }
        if (
            not markets
            or {m["spec"].rate_direction for m in markets} != {"cut", "hold", "hike"}
            or any(m["parent"].get("negRisk") is not True for m in markets)
        ):
            return {
                **self._missing("rate_partition_not_declared_or_incomplete"),
                "probabilities": None,
            }
        for offset, a in enumerate(markets):
            for b in markets[offset + 1 :]:
                left, right = a["spec"], b["spec"]
                low = max(
                    left.lower if left.lower is not None else -math.inf,
                    right.lower if right.lower is not None else -math.inf,
                )
                high = min(
                    left.upper if left.upper is not None else math.inf,
                    right.upper if right.upper is not None else math.inf,
                )

                def contains(spec, value):
                    return (
                        spec.lower is None
                        or value > spec.lower
                        or value == spec.lower
                        and spec.lower_inclusive
                    ) and (
                        spec.upper is None
                        or value < spec.upper
                        or value == spec.upper
                        and spec.upper_inclusive
                    )

                if low < high or low == high and contains(left, low) and contains(right, low):
                    return {
                        **self._missing("overlapping_rate_bins"),
                        "status": "unsupported",
                        "probabilities": None,
                    }
        common = None
        for market in markets:
            times = set(market["quotes"].get("Yes", {})) & set(market["quotes"].get("No", {}))
            times = {value for value in times if value <= origin}
            common = times if common is None else common & times
        if not common:
            return {**self._missing("no_common_rate_bin_asof"), "probabilities": None}
        at = max(common)  # Never search older buckets for a more convenient/coherent sum.
        probabilities, references, components = {"cut": 0.0, "hold": 0.0, "hike": 0.0}, [], []
        for market in markets:
            pair, reason = self._quote_pair(market, origin, at)
            if pair is None:
                return {**self._missing(reason), "probabilities": None}
            probabilities[market["spec"].rate_direction] += pair[0]["probability"]
            references.append(self._reference(market, pair, origin))
            components.append(
                {"specification": asdict(market["spec"]), "yes_probability": pair[0]["probability"]}
            )
        total = sum(probabilities.values())
        if any(not 0 <= value <= 1 for value in probabilities.values()):
            return {
                "status": "incoherent",
                "reason_codes": ["rate_direction_probability_outside_0_to_1"],
                "probabilities": None,
                "observed_sum": total,
                "source_refs": references,
                "historical_pit_verified": False,
            }
        if abs(total - 1) > 0.05 + 1e-12:
            return {
                "status": "incoherent",
                "reason_codes": ["rate_bin_sum_outside_0_95_to_1_05"],
                "probabilities": None,
                "observed_sum": total,
                "source_refs": references,
                "historical_pit_verified": False,
            }
        return {
            "status": "available",
            "reason_codes": [],
            "probabilities": probabilities,
            "observed_sum": total,
            "normalized": False,
            "components": components,
            "source_refs": references,
            "historical_pit_verified": False,
        }

    def lookup(self, origin: datetime) -> Dict[str, Any]:
        if not isinstance(origin, datetime) or origin.tzinfo is None or origin.utcoffset() is None:
            raise ValueError("Historical lookup origin requires an aware datetime.")
        origin = origin.astimezone(timezone.utc)
        channels = {}
        for channel, families in FAMILIES.items():
            candidates = []
            for group_id, markets in self._group_markets.items():
                if self._plans[group_id]["channel"] != channel:
                    continue
                known = [m for m in markets if m["valid_from"] <= origin]
                if not known:
                    continue
                end = min(_utc(m["row"]["source_end_date"]) for m in known)
                if origin < end <= origin + timedelta(days=21):
                    candidates.append((end, group_id, markets))
            if not candidates:
                channels[channel] = self._missing("no_created_upcoming_group_within_21_days")
                continue
            end, group_id, markets = min(candidates, key=lambda item: (item[0], item[1]))
            context = {
                "group_id": group_id,
                "declared_event_end": end.isoformat(),
                "days_to_declared_end": (end - origin).total_seconds() / 86400,
                "end_time_verified_as_release": False,
            }
            quarantined = [m for m in markets if m["row"].get("quarantine_reasons")]
            eligible = [m for m in markets if not m["row"].get("quarantine_reasons")]
            if channel == "fed":
                result = (
                    self._fed(eligible, origin)
                    if not quarantined
                    else self._missing("quarantined_rate_bins")
                )
            else:
                measurements = {
                    family: self._measurement(eligible, family, origin) for family in families
                }
                available = any(value["status"] == "available" for value in measurements.values())
                result = {
                    "status": "available" if available else "missing",
                    "measurements": measurements,
                    "reason_codes": [] if available else ["no_available_reference_measurement"],
                    "source_refs": [
                        ref for value in measurements.values() for ref in value["source_refs"]
                    ],
                    "quarantined_market_ids": sorted({m["row"]["market_id"] for m in quarantined}),
                    "historical_pit_verified": False,
                }
            channels[channel] = {**context, **result}
        return copy.deepcopy(
            {
                "feature_version": FEATURE_VERSION,
                "origin": origin.isoformat(),
                "source_manifest_sha256": self.manifest_sha256,
                "channels": channels,
                "data_mode": "reconstructed_history",
                "historical_pit_verified": False,
                "limitations": [
                    "Current metadata does not verify historical publications or rules.",
                    "Bucket-end availability proxy; observations older than 24 hours are withheld.",
                    "No stock probability, fitted weight or outcome label is inferred.",
                ],
            }
        )
