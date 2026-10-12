"""Oracle3 MCP server.

Exposes public Kalshi and Polymarket market data, the venues' published fee
schedules, static no-arbitrage checks across related contracts, and a local
paper-trading ledger to any Model Context Protocol client.

No tool can place a real order: the server imports no authenticated trader and
``paper_order`` only writes to a local JSON ledger.

Run it over stdio (the default, for desktop and IDE clients)::

    oracle3-mcp

or over streamable HTTP::

    oracle3-mcp --transport streamable-http
"""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
import math
from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, Any, Literal

import httpx
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from oracle3 import __version__
from oracle3.arbitrage import RELATIONS, Quote
from oracle3.arbitrage import check_constraint as _check
from oracle3.fees import KalshiSchedule, PolymarketSchedule, UnsupportedFeeSchedule
from oracle3.mcp_server import venues
from oracle3.mcp_server.paper import PaperLedger
from oracle3.pricing.distortion import ProbitDistortion

Venue = Literal['kalshi', 'polymarket']
Relation = Literal[
    'implication', 'exclusivity', 'complement', 'same_event', 'event_sum'
]

# Parameter types shared by several tools; the descriptions reach the tool schemas.
VenueArg = Annotated[Venue, Field(description="Venue: 'kalshi' or 'polymarket'.")]
MarketIdArg = Annotated[
    str,
    Field(
        description=(
            'Kalshi market ticker (e.g. KXFEDDECISION-28JAN-H0) or Polymarket market '
            'id (e.g. 2589812), as returned by search_markets.'
        )
    ),
]
RelationArg = Annotated[
    Relation,
    Field(
        description=(
            'Bound to test: implication (A implies B, so P(A) <= P(B)), exclusivity '
            '(at most one outcome happens), complement (exactly one of two), '
            'same_event (one event quoted on two venues) or event_sum (exactly one '
            'of n). See list_relation_types.'
        )
    ),
]
ContractsArg = Annotated[
    float,
    Field(
        description='Finite, positive contracts bought on every leg (default 1). Fees scale with size.'
    ),
]
MakerArg = Annotated[
    bool,
    Field(
        description=(
            'Price fees as resting maker orders instead of taker orders (default '
            'false). Maker fees are lower or zero, but a resting order may not fill.'
        )
    ),
]

INSTRUCTIONS = """\
Oracle3 gives read access to Kalshi and Polymarket and checks whether prices of
related contracts violate the axioms of probability after venue fees.

Prices are dollars per contract, between 0 and 1. No tool places a real order;
paper_order fills against the live order book into a local paper ledger.

Typical flow: search_markets -> get_quote or get_orderbook on candidates ->
check_constraint_live with the relation you believe holds -> paper_order the
legs of the best basket.

Relations: implication (A implies B), exclusivity (at most one outcome),
complement (exactly one of two), same_event (one event on two venues),
event_sum (exactly one of n). The checks calculate price bounds conditional on
the supplied relation; they do not verify resolution rules, order-book depth
or full fills. Results explicitly mark relation_verified and execution_verified
false. Read both markets' resolution rules before relying on a result.

Fees follow each market's own schedule from the venue API: Kalshi taker
0.07*M*C*P*(1-P) (maker 0.0175*M*C*P*(1-P) only on series with maker fees),
Polymarket taker rate*C*p*(1-p), Polymarket makers free.
"""

WEBSITE = 'https://yichengyang-ethan.github.io/oracle3-prediction-market-agent/'

try:  # mcp >= 2 renamed FastMCP to MCPServer and takes the version directly
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    mcp: Any = MCPServer(
        'oracle3', instructions=INSTRUCTIONS, website_url=WEBSITE, version=__version__
    )
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined]
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore

    mcp = FastMCP('oracle3', instructions=INSTRUCTIONS, website_url=WEBSITE)
    # FastMCP has no version argument; without this the SDK's own version is reported.
    mcp._mcp_server.version = __version__

# One INFO line per venue request drowns out the protocol traffic on stderr.
logging.getLogger('httpx').setLevel(logging.WARNING)

# Errors an agent can act on: a venue call failed, an input was out of range, or a
# market uses a fee schedule we cannot price. mcp 2.x hides the message of any
# exception that is not a ToolError, so these are re-raised as ToolError.
_USER_ERRORS = (
    venues.VenueError,
    UnsupportedFeeSchedule,
    ValueError,
    KeyError,
    ArithmeticError,  # e.g. decimal.InvalidOperation for a NaN size
    httpx.InvalidURL,  # a market id that cannot form a URL
)


def _agent_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except _USER_ERRORS as exc:
                raise ToolError(str(exc)) from exc

        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except _USER_ERRORS as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


READ = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
PURE = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
LOCAL_READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
PAPER = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
PAPER_RESET = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, openWorldHint=False
)


class QuoteInput(BaseModel):
    """One contract's prices in dollars per contract. A missing ask is derived from the opposite side's bid."""

    market_id: str = Field(
        description='Label for this contract in the result, such as its ticker or market id.'
    )
    venue: Venue = Field(
        default='kalshi',
        description="Venue whose fee schedule applies: 'kalshi' (default) or 'polymarket'.",
    )
    yes_bid: float | None = Field(
        default=None, description='Best YES bid in dollars (0-1).'
    )
    yes_ask: float | None = Field(
        default=None,
        description='Best YES ask in dollars (0-1). If omitted, 1 - no_bid is used.',
    )
    no_bid: float | None = Field(
        default=None, description='Best NO bid in dollars (0-1).'
    )
    no_ask: float | None = Field(
        default=None,
        description='Best NO ask in dollars (0-1). If omitted, 1 - yes_bid is used.',
    )
    kalshi_multiplier: float | None = Field(
        default=None,
        description='Kalshi series fee multiplier M, from get_quote (default 1).',
    )
    kalshi_maker_fees: bool = Field(
        default=False,
        description='Whether the Kalshi series charges maker fees, from get_quote.',
    )
    polymarket_rate: float | None = Field(
        default=None,
        description='Polymarket taker fee rate for this market (default: 0.07, the highest documented rate).',
    )


class MarketRef(BaseModel):
    """A live market to check, identified by venue and market id."""

    venue: Venue = Field(description="'kalshi' or 'polymarket'.")
    market_id: str = Field(
        description='Kalshi ticker or Polymarket market id, as returned by search_markets.'
    )


def _schedule(q: QuoteInput) -> KalshiSchedule | PolymarketSchedule:
    if q.venue == 'kalshi':
        return KalshiSchedule(
            multiplier=Decimal(
                str(1.0 if q.kalshi_multiplier is None else q.kalshi_multiplier)
            ),
            maker_fees=q.kalshi_maker_fees,
        )
    if q.polymarket_rate is None:
        return PolymarketSchedule()
    return PolymarketSchedule(
        rate=Decimal(str(q.polymarket_rate)), source='supplied rate'
    )


def _quote_dict(q: Quote) -> dict[str, Any]:
    schedule = q.fee_schedule()
    return {
        'venue': q.venue,
        'market_id': q.market_id,
        'title': q.title,
        'yes_bid': q.yes_bid,
        'yes_ask': q.ask('yes'),
        'no_bid': q.no_bid,
        'no_ask': q.ask('no'),
        'fee_schedule': schedule.describe(),
    }


@mcp.tool(annotations=READ)
@_agent_errors
async def search_markets(
    venue: VenueArg,
    query: Annotated[
        str,
        Field(
            description=(
                'Case-insensitive keywords matched against event titles, market titles '
                'and (on Kalshi) tickers. Empty lists open markets without a keyword filter.'
            )
        ),
    ] = '',
    limit: Annotated[
        int,
        Field(description='Maximum markets to return, 1-100 (clamped). Default 20.'),
    ] = 20,
    series_ticker: Annotated[
        str,
        Field(
            description=(
                'Kalshi only: list the open markets of this series (e.g. KXFEDDECISION) '
                'instead of searching by keyword; query is then ignored. Ignored on Polymarket.'
            )
        ),
    ] = '',
) -> dict[str, Any]:
    """Search open Kalshi or Polymarket markets and return candidates with their best prices.

    Start here to find market ids, then call get_quote or get_orderbook for live
    prices, or get_market for resolution details. Reads the venues' public APIs,
    so no account or API key is needed; Kalshi keyword search scans up to 1,000
    open events. Returns {venue, query, count, markets}; each market has
    market_id, title, best bid and ask in dollars, volume and closing date.
    """
    markets = await venues.search(venue, query, max(1, min(limit, 100)), series_ticker)
    return {'venue': venue, 'query': query, 'count': len(markets), 'markets': markets}


@mcp.tool(annotations=READ)
@_agent_errors
async def get_market(venue: VenueArg, market_id: MarketIdArg) -> dict[str, Any]:
    """Get one market's details: title, best bid and ask, volume and closing date.

    Kalshi results include the resolution rules (rules_primary, rules_secondary);
    Polymarket results include the outcomes, token ids, neg-risk flag and the
    market's fee schedule. Read the rules of every leg here before trusting a
    relation in check_constraint_live. For prices plus fee terms only, use
    get_quote; for depth, use get_orderbook. Public data, no API key.
    """
    return await venues.market(venue, market_id)


@mcp.tool(annotations=READ)
@_agent_errors
async def get_orderbook(
    venue: VenueArg,
    market_id: MarketIdArg,
    depth: Annotated[
        int, Field(description='Price levels per side, 1-50 (clamped). Default 10.')
    ] = 10,
) -> dict[str, Any]:
    """Get the displayed order book for YES and NO as [price, size] levels, best first.

    Prices are dollars per contract and sizes are contracts. Use it to see how
    much of a basket can fill near the quoted price before paper_order; use
    get_quote when only the top of book matters. Kalshi publishes bids only, so
    each side's asks are derived from the other side's bids (a YES ask of p is a
    NO bid of 1 - p). Public data, no API key.
    """
    return await venues.orderbook(venue, market_id, max(1, min(depth, 50)))


@mcp.tool(annotations=READ)
@_agent_errors
async def get_quote(venue: VenueArg, market_id: MarketIdArg) -> dict[str, Any]:
    """Get the best bid and ask on YES and NO, in dollars, plus the market's fee schedule.

    Use it to collect prices for check_constraint, or to read the fee terms
    (Kalshi multiplier and maker fees, Polymarket taker rate) that trading_fee
    takes. check_constraint_live fetches quotes itself, so this is not needed
    before calling it. A missing ask is derived from the opposite side's bid.
    Returns {venue, market_id, title, yes_bid, yes_ask, no_bid, no_ask,
    fee_schedule}. Public data, no API key.
    """
    return _quote_dict(await venues.quote(venue, market_id))


@mcp.tool(annotations=PURE)
@_agent_errors
def trading_fee(
    venue: VenueArg,
    price: Annotated[
        float,
        Field(
            description='Fill price in dollars per contract, strictly between 0 and 1.'
        ),
    ],
    contracts: Annotated[
        float, Field(description='Contracts in the fill; must be positive.')
    ],
    maker: Annotated[
        bool,
        Field(
            description='Price a resting maker order instead of a taker order (default false).'
        ),
    ] = False,
    kalshi_multiplier: Annotated[
        float,
        Field(
            description=(
                "Kalshi series fee multiplier M, from get_quote's fee_schedule "
                '(default 1). Ignored for Polymarket.'
            )
        ),
    ] = 1.0,
    kalshi_maker_fees: Annotated[
        bool,
        Field(
            description=(
                'Whether the Kalshi series charges maker fees, from get_quote '
                '(default false). Ignored for Polymarket.'
            )
        ),
    ] = False,
    polymarket_rate: Annotated[
        float | None,
        Field(
            description=(
                'Polymarket taker fee rate for the market, from get_quote. Defaults to '
                '0.07, the highest documented rate. Ignored for Kalshi.'
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Compute the venue fee for one fill at a given price and size; a pure calculation with no network calls.

    Kalshi charges round_up(0.07 x M x C x P x (1 - P)) on taker fills, and
    0.0175 x M x C x P x (1 - P) on maker fills where the series has maker fees;
    Polymarket charges takers rate x C x p x (1 - p) and makers nothing. Use it
    to price a single leg or a hypothetical fill; take a live market's fee terms
    from get_quote, and use check_constraint to price a whole basket. Returns
    fee, fee_per_contract, order_type and the schedule used.
    """
    q = QuoteInput(
        market_id='-',
        venue=venue,
        kalshi_multiplier=kalshi_multiplier,
        kalshi_maker_fees=kalshi_maker_fees,
        polymarket_rate=polymarket_rate,
    )
    schedule = _schedule(q)
    fee = schedule.fee(price, Decimal(str(contracts)), maker=maker)
    return {
        'fee': float(fee),
        'fee_per_contract': float(fee) / contracts,
        'order_type': 'maker' if maker else 'taker',
        'schedule': schedule.describe(),
    }


@mcp.tool(annotations=PURE)
@_agent_errors
def check_constraint(
    relation: RelationArg,
    quotes: Annotated[
        list[QuoteInput],
        Field(
            description=(
                'One quote per contract, in the order the relation expects: '
                'implication (A, B) where A implies B; complement and same_event (A, B); '
                'exclusivity and event_sum every outcome.'
            )
        ),
    ],
    contracts: ContractsArg = 1.0,
    maker: MakerArg = False,
) -> dict[str, Any]:
    """Calculate a price-bound violation and basket economics conditional on the relation you supply.

    A pure calculation with no network calls: use it for hypothetical or
    historical prices, and use check_constraint_live to fetch current quotes
    instead. Returns violated, profitable_after_fees, the best basket (legs,
    cost, conditional payoff bound, fees, gross and net edge), every candidate basket,
    any missing quotes, and the assumptions behind the check (full fills at the
    quoted prices, a correctly specified relation). The assessment object marks
    relation_verified=false, execution_verified=false and conditional_on_relation=true:
    this tool does not verify resolution rules, order-book depth or full fills.
    Quote prices must be finite and in [0, 1]; prices used to calculate fees must
    be strictly between 0 and 1. Each (venue, market_id) must be unique.
    """
    parsed = [
        Quote(
            market_id=q.market_id,
            venue=q.venue,
            yes_bid=q.yes_bid,
            yes_ask=q.yes_ask,
            no_bid=q.no_bid,
            no_ask=q.no_ask,
            schedule=_schedule(q),
        )
        for q in quotes
    ]
    return _check(relation, parsed, contracts=contracts, maker=maker).to_dict()


@mcp.tool(annotations=READ)
@_agent_errors
async def check_constraint_live(
    relation: RelationArg,
    markets: Annotated[
        list[MarketRef],
        Field(
            description=(
                'Markets to check, each {venue, market_id}, in the order the relation '
                'expects: two for implication (A then B), complement and same_event; '
                'every outcome for exclusivity and event_sum. Venues may be mixed.'
            )
        ),
    ],
    contracts: ContractsArg = 1.0,
    maker: MakerArg = False,
) -> dict[str, Any]:
    """Fetch live quotes and fees, then calculate price bounds conditional on the supplied relation.

    This is the main research call: find related contracts with search_markets,
    confirm with get_market that their resolution rules really satisfy the
    relation, then pass them here. Read-only: it calls the venues' public APIs
    and places no orders. Returns the check_constraint result (violated,
    profitable_after_fees, best basket with net edge) plus the quotes it used;
    to simulate the trade, call paper_order for each leg of the best basket.
    Live quotes do not verify resolution rules, order-book depth or full fills:
    assessment.relation_verified and assessment.execution_verified remain false.
    """
    # Reject invalid sizes and repeated/blank identities before fetching quotes.
    _check(
        relation,
        [Quote(market_id=m.market_id, venue=m.venue) for m in markets],
        contracts=contracts,
        maker=maker,
    )
    quotes = [await venues.quote(m.venue, m.market_id) for m in markets]
    result = _check(relation, quotes, contracts=contracts, maker=maker).to_dict()
    result['quotes'] = [_quote_dict(q) for q in quotes]
    return result


@mcp.tool(annotations=PURE)
@_agent_errors
def list_relation_types() -> dict[str, Any]:
    """List the relation types the constraint checks support and the probability bound each enforces.

    Static reference data with no network or file access, for example
    implication: P(A) <= P(B). Use it to choose the relation argument of
    check_constraint or check_constraint_live; to see concrete market pairs
    saved on this machine, use list_relations instead. Returns
    {relations: {name: bound}}.
    """
    return {'relations': RELATIONS}


@mcp.tool(annotations=PURE)
@_agent_errors
def fair_value(
    market_price: Annotated[
        float,
        Field(description='Observed YES price in dollars, strictly between 0 and 1.'),
    ],
    lam: Annotated[
        float,
        Field(
            description=(
                'Pricing-wedge parameter lambda of the Wang transform. Positive values '
                'mean prices sit above the true probability, most of all for longshots. '
                'Default 0.183.'
            )
        ),
    ] = 0.183,
) -> dict[str, Any]:
    """Convert a market price into the probability it implies under the Wang transform, and report the premium.

    Solves p_mkt = Phi(Phi^-1(p) + lam) for p; a pure calculation with no network
    calls. Use it to strip the favourite-longshot premium from a quoted price
    before comparing it with your own forecast. The default lam = 0.183 is the
    pooled estimate in Yang (2026), SSRN 6468338; it pools real-money and
    play-money venues, so treat it as illustrative. Returns market_price,
    lambda, implied_probability and premium (price minus implied probability).
    """
    if not 0.0 < market_price < 1.0:
        raise ValueError('market_price must be strictly between 0 and 1')
    if not math.isfinite(lam):
        raise ValueError('lam must be finite')
    implied = ProbitDistortion(lam).inverse(market_price)
    return {
        'market_price': market_price,
        'lambda': lam,
        'implied_probability': implied,
        'premium': market_price - implied,
    }


@mcp.tool(annotations=LOCAL_READ)
@_agent_errors
def list_relations(
    market_id: Annotated[
        str,
        Field(
            description=(
                'Only relations that include this market id (Kalshi ticker or '
                'Polymarket id). Empty returns all.'
            )
        ),
    ] = '',
    spread_type: Annotated[
        str,
        Field(
            description=(
                'Only this relation type: same_event, cross_platform, implication, '
                'exclusivity, conditional, structural, cointegration or complement. '
                'Empty returns all.'
            )
        ),
    ] = '',
    status: Annotated[
        str,
        Field(
            description=(
                'Only this lifecycle status: discovered, validated, deployed, retired '
                'or invalidated. Empty returns all.'
            )
        ),
    ] = '',
) -> dict[str, Any]:
    """List the market relations saved on this machine by the oracle3 research CLI, optionally filtered.

    Reads ~/.oracle3/relations.json with no network calls. Use it to reuse pairs
    already discovered or validated offline, then pass a pair to
    check_constraint_live. For the abstract relation kinds and their bounds,
    call list_relation_types instead. Returns the store path, a count and the
    matching relations (markets, type, status and validation details); the list
    is empty if the CLI has saved none.
    """
    from oracle3.market.relations import RelationStore

    store = RelationStore()
    rows = store.find_by_market(market_id) if market_id else store.list()
    if spread_type:
        rows = [r for r in rows if r.spread_type == spread_type]
    if status:
        rows = [r for r in rows if r.status == status]
    return {
        'store': str(store.path),
        'count': len(rows),
        'relations': [r.to_dict() for r in rows],
    }


@mcp.tool(annotations=PAPER)
@_agent_errors
async def paper_order(
    venue: VenueArg,
    market_id: MarketIdArg,
    side: Annotated[
        Literal['yes', 'no'],
        Field(description="Side of the binary contract to buy: 'yes' or 'no'."),
    ],
    contracts: Annotated[
        float, Field(description='Contracts to buy; must be finite and positive.')
    ],
    limit_price: Annotated[
        float,
        Field(
            description=(
                'Highest price per contract to pay, in dollars, strictly between 0 '
                'and 1; ask levels above it are skipped.'
            )
        ),
    ],
    request_id: Annotated[
        str | None,
        Field(
            description='Optional nonblank retry key, unique until paper_reset. Reuse only for the identical order intent.'
        ),
    ] = None,
) -> dict[str, Any]:
    """Simulate buying YES or NO contracts in the local paper ledger, filling against the live order book with venue fees. Never sends an order to a venue.

    Walks the displayed asks from the best price up to limit_price and charges
    the venue fee on each level. Use it to test the legs of a basket found by
    check_constraint_live. Writes ~/.oracle3/mcp_paper_ledger.json (or the path
    in ORACLE3_MCP_LEDGER); only buys are supported and positions are held at
    cost. Returns status (filled, partially_filled, unfilled, or rejected when
    paper cash is short), requested and filled contracts, average price, cost,
    fees and the per-level fills. Supply request_id for safe retries: an identical
    venue/market/side/quantity/limit returns the original result without another
    fill, including unfilled or rejected results; changed intent is an error.
    Without request_id, each call is a new order. Reset clears retry history.
    """
    intent = PaperLedger.order_intent(
        venue=venue,
        market_id=market_id,
        side=side,
        contracts=contracts,
        limit_price=limit_price,
    )
    ledger = PaperLedger()
    prior = ledger.prior_result(request_id, intent)
    if prior is not None:
        return prior
    book = await venues.orderbook(venue, market_id, depth=50)
    schedule = await venues.fee_schedule(venue, market_id)
    return ledger.buy(
        venue=venue,
        market_id=market_id,
        side=side,
        contracts=contracts,
        limit_price=limit_price,
        asks=[(float(p), float(q)) for p, q in book[side]['asks']],
        schedule=schedule,
        request_id=request_id,
    )


@mcp.tool(annotations=LOCAL_READ)
@_agent_errors
def paper_portfolio() -> dict[str, Any]:
    """Show the local paper ledger: cash, open positions and the number of fills.

    Reads ~/.oracle3/mcp_paper_ledger.json (or ORACLE3_MCP_LEDGER) with no
    network calls. Use it after paper_order to confirm fills and remaining cash.
    Starting cash is $10,000; positions are carried at cost, and the ledger does
    not mark to market or track resolution. Returns the ledger path,
    initial_cash, cash, positions and fills.
    """
    return PaperLedger().portfolio()


@mcp.tool(annotations=PAPER_RESET)
@_agent_errors
def paper_reset(
    confirm: Annotated[
        bool,
        Field(
            description='Must be true to erase the ledger; otherwise nothing changes.'
        ),
    ] = False,
) -> dict[str, Any]:
    """Erase the local paper ledger and restore the $10,000 starting cash.

    Deletes all paper positions, fill history and request_id retry history. No
    venue is contacted and real accounts are never touched. Without confirm=true it returns status
    not_reset and changes nothing. Use it to start a fresh paper-trading
    session.
    """
    if not confirm:
        return {
            'status': 'not_reset',
            'reason': 'pass confirm=true to erase the paper ledger',
        }
    return PaperLedger().reset()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='oracle3-mcp', description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        '--transport', choices=['stdio', 'streamable-http', 'sse'], default='stdio'
    )
    args = parser.parse_args(argv)
    mcp.run(transport=args.transport)


if __name__ == '__main__':
    main()
