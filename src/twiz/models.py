from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RawItem:
    source: str
    source_type: str
    url: str
    published_at: str
    ticker_candidates: List[str]
    title: str
    text: str
    engagement: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SentimentSignal:
    ticker: str
    label: str
    score: float
    confidence: float
    weight: float
    model: str
    source_refs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriceLevel:
    price: float
    kind: str
    touches: int
    distance_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TechnicalSnapshot:
    ticker: str
    timeframe: str
    levels: List[Dict[str, Any]]
    indicators: Dict[str, Optional[float]]
    regime: str
    chart_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    strategy_id: str
    ticker: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    equity_curve_path: Optional[str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptionContract:
    symbol: str
    option_type: str
    strike: float
    expiration: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptionSnapshot:
    ticker: str
    expiration: Optional[str]
    contracts: List[Dict[str, Any]]
    greeks: Dict[str, Optional[float]]
    skew_summary: Dict[str, Optional[float]]
    payoff_scenarios: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportBundle:
    date: str
    watchlist_table: List[Dict[str, Any]]
    macro_radar: List[Dict[str, Any]]
    per_ticker_notes: Dict[str, List[str]]
    sentiment_deltas: Dict[str, Dict[str, float]]
    technical_setup: Dict[str, Dict[str, Any]]
    options_context: Dict[str, Dict[str, Any]]
    risk_flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
