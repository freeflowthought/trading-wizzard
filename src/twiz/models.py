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
    last_trade_at: Optional[str] = None

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
    full_chain_contracts: List[Dict[str, Any]] = field(default_factory=list)
    chain_metadata: Dict[str, Any] = field(default_factory=dict)

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
    events: List[Dict[str, Any]] = field(default_factory=list)
    playbook: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    macro_gate: Dict[str, Any] = field(default_factory=dict)
    leadership: Dict[str, Any] = field(default_factory=dict)
    ipo_watch: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fundamentals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    daily_brief: Dict[str, Any] = field(default_factory=dict)
    prediction_markets: List[Dict[str, Any]] = field(default_factory=list)
    action_plan: Dict[str, Any] = field(default_factory=dict)
    cross_market: List[Dict[str, Any]] = field(default_factory=list)
    shunge_context: Dict[str, Any] = field(default_factory=dict)
    market_structure: Dict[str, Any] = field(default_factory=dict)
    scenario_context: Dict[str, Any] = field(default_factory=dict)
    analyst_study: Dict[str, Any] = field(default_factory=dict)
    analysis_run: Dict[str, Any] = field(default_factory=dict)
    directional_context: Dict[str, Any] = field(default_factory=dict)
    cloud_capex_context: Dict[str, Any] = field(default_factory=dict)
    portfolio_plan: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
