"""Static no-arbitrage checks for related binary event contracts.

Related contracts are bound together by the axioms of probability. When quoted
prices break one of those bounds, a basket of contracts that pays at least a
known amount in every state can be bought for less than that amount, provided
the stated relation holds. For each supported relation this module calculates
candidate baskets at the supplied prices, their conditional payoff bounds,
the venue fee on every leg, and the edge left after fees. It does not verify
resolution rules or available order-book depth.

Relations (quotes are passed in the order shown):

``implication`` (A, B)
    A implies B, so P(A) <= P(B). Basket: NO on A + YES on B, which pays at
    least 1 (and 2 if B happens without A).
``exclusivity`` (A1, ..., An)
    At most one of the outcomes happens, so the probabilities sum to at most 1.
    Basket: NO on every outcome, which pays at least n - 1.
``complement`` (A, B)
    Exactly one of A and B happens, so P(A) + P(B) = 1. Baskets: YES on both
    or NO on both; each pays exactly 1.
``same_event`` (A, B)
    A and B are the same event quoted on two venues, so P(A) = P(B). Baskets:
    YES on A + NO on B, or NO on A + YES on B; each pays exactly 1.
``event_sum`` (A1, ..., An)
    Exactly one outcome happens, so the probabilities sum to 1. Baskets: YES on
    every outcome (pays 1) or NO on every outcome (pays n - 1).

Prices are dollars per contract. A missing ask is derived from the opposite
side's bid (NO ask = 1 - YES bid), which is how both venues link the two sides
of a binary market.

The checks are static. They assume every leg fills at the quoted price for the
requested size and that the relation is specified correctly, including
matching resolution rules across venues. Partial fills, legging risk and the
cost of capital locked until resolution are not modeled.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Literal

from oracle3.fees import FeeSchedule, default_schedule

__all__ = [
    'RELATIONS',
    'Basket',
    'ConstraintCheck',
    'Leg',
    'Quote',
    'check_constraint',
]

Side = Literal['yes', 'no']

RELATIONS: dict[str, str] = {
    'implication': 'P(A) <= P(B)',
    'exclusivity': 'sum_i P(A_i) <= 1',
    'complement': 'P(A) + P(B) = 1',
    'same_event': 'P(A) = P(B)',
    'event_sum': 'sum_i P(A_i) = 1',
}

ASSUMPTIONS = (
    'every leg fills at the quoted price for the full size',
    'the relation is specified correctly, including matching resolution rules',
    'no settlement fee (neither Kalshi nor Polymarket charges one)',
    'capital locked until resolution is not charged a cost',
)


def _clean(value: float) -> float:
    return round(value, 6)


@dataclass
class Quote:
    """Best bid and ask for both sides of one binary contract."""

    market_id: str
    venue: str = 'kalshi'
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    schedule: FeeSchedule | None = None
    title: str = ''

    def ask(self, side: Side) -> float | None:
        """Price to buy one contract of ``side``, or ``None`` if unquoted."""
        if side == 'yes':
            if self.yes_ask is not None:
                return self.yes_ask
            return None if self.no_bid is None else _clean(1.0 - self.no_bid)
        if self.no_ask is not None:
            return self.no_ask
        return None if self.yes_bid is None else _clean(1.0 - self.yes_bid)

    def fee_schedule(self) -> FeeSchedule:
        return (
            self.schedule if self.schedule is not None else default_schedule(self.venue)
        )


@dataclass
class Leg:
    market_id: str
    venue: str
    side: Side
    price: float
    contracts: float
    cost: float
    fee: float


@dataclass
class Basket:
    """One candidate basket and its economics at the requested size."""

    description: str
    legs: list[Leg]
    payoff_per_contract: float
    contracts: float
    cost: float
    fees: float
    gross_edge: float
    net_edge: float

    @property
    def gross_edge_per_contract(self) -> float:
        return _clean(self.gross_edge / self.contracts)

    @property
    def net_edge_per_contract(self) -> float:
        return _clean(self.net_edge / self.contracts)

    @property
    def fees_per_contract(self) -> float:
        return _clean(self.fees / self.contracts)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out['gross_edge_per_contract'] = self.gross_edge_per_contract
        out['net_edge_per_contract'] = self.net_edge_per_contract
        out['fees_per_contract'] = self.fees_per_contract
        return out


@dataclass
class ConstraintCheck:
    relation: str
    constraint: str
    contracts: float
    maker: bool
    baskets: list[Basket] = field(default_factory=list)
    missing_quotes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Basket | None:
        if not self.baskets:
            return None
        return max(self.baskets, key=lambda b: (b.net_edge, b.gross_edge))

    @property
    def violated(self) -> bool:
        """True if some basket costs less than its guaranteed payoff."""
        return any(b.gross_edge > 0 for b in self.baskets)

    @property
    def profitable_after_fees(self) -> bool:
        best = self.best
        return best is not None and best.net_edge > 0

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        return {
            'relation': self.relation,
            'constraint': self.constraint,
            'contracts': self.contracts,
            'order_type': 'maker' if self.maker else 'taker',
            'violated': self.violated,
            'profitable_after_fees': self.profitable_after_fees,
            'best': best.to_dict() if best else None,
            'baskets': [b.to_dict() for b in self.baskets],
            'missing_quotes': self.missing_quotes,
            'assessment': {
                'relation_verified': False,
                'execution_verified': False,
                'conditional_on_relation': True,
            },
            'assumptions': list(ASSUMPTIONS),
        }


def _candidates(relation: str, n: int) -> list[tuple[str, list[tuple[int, Side]], int]]:
    """Baskets as (description, legs, guaranteed payoff per contract)."""
    all_yes: list[tuple[int, Side]] = [(i, 'yes') for i in range(n)]
    all_no: list[tuple[int, Side]] = [(i, 'no') for i in range(n)]
    if relation == 'implication':
        return [('NO on A + YES on B', [(0, 'no'), (1, 'yes')], 1)]
    if relation == 'exclusivity':
        return [('NO on every outcome', all_no, n - 1)]
    if relation == 'complement':
        return [
            ('YES on A + YES on B', all_yes, 1),
            ('NO on A + NO on B', all_no, 1),
        ]
    if relation == 'same_event':
        return [
            ('YES on A + NO on B', [(0, 'yes'), (1, 'no')], 1),
            ('NO on A + YES on B', [(0, 'no'), (1, 'yes')], 1),
        ]
    if relation == 'event_sum':
        return [
            ('YES on every outcome', all_yes, 1),
            ('NO on every outcome', all_no, n - 1),
        ]
    raise ValueError(
        f'unknown relation {relation!r}; expected one of {sorted(RELATIONS)}'
    )


def _validate_quotes(quotes: list[Quote]) -> None:
    """Validate every supplied field before candidate baskets can skip any leg."""
    identities: set[tuple[str, str]] = set()
    for quote in quotes:
        if not isinstance(quote.market_id, str) or not quote.market_id.strip():
            raise ValueError('market_id must be a nonblank string')
        if not isinstance(quote.venue, str) or not quote.venue.strip():
            raise ValueError('venue must be a nonblank string')
        identity = (quote.venue.strip().lower(), quote.market_id.strip())
        if identity in identities:
            raise ValueError(f'duplicate contract: {identity[0]}:{identity[1]}')
        identities.add(identity)
        # Validate even unused fields and quotes after an earlier missing leg.
        for name in ('yes_bid', 'yes_ask', 'no_bid', 'no_ask'):
            price = getattr(quote, name)
            if price is not None and (not math.isfinite(price) or not 0 <= price <= 1):
                raise ValueError(
                    f'{quote.venue}:{quote.market_id} {name} must be finite '
                    'and between 0 and 1'
                )


def check_constraint(
    relation: str,
    quotes: list[Quote],
    *,
    contracts: float = 1.0,
    maker: bool = False,
) -> ConstraintCheck:
    """Calculate price bounds conditional on the supplied ``relation``.

    This does not verify the relation's resolution rules, order-book depth or
    whether the basket can fill. All supplied prices must be finite and in
    [0, 1]; prices used for fee calculations must be strictly between 0 and 1.

    Args:
        relation: One of :data:`RELATIONS`.
        quotes: Quotes in the order the relation expects (see module docstring).
        contracts: Contracts bought on every leg.
        maker: Price fees as resting (maker) orders instead of taker orders.
            Maker legs usually pay less but are not guaranteed to fill.
    """
    if relation not in RELATIONS:
        raise ValueError(
            f'unknown relation {relation!r}; expected one of {sorted(RELATIONS)}'
        )
    pairwise = relation in {'implication', 'complement', 'same_event'}
    if pairwise and len(quotes) != 2:
        raise ValueError(f'{relation} needs exactly 2 quotes, got {len(quotes)}')
    if not pairwise and len(quotes) < 2:
        raise ValueError(f'{relation} needs at least 2 quotes, got {len(quotes)}')
    if not math.isfinite(contracts) or contracts <= 0:
        raise ValueError('contracts must be finite and positive')
    _validate_quotes(quotes)

    result = ConstraintCheck(
        relation=relation,
        constraint=RELATIONS[relation],
        contracts=contracts,
        maker=maker,
    )
    size = Decimal(str(contracts))
    for description, legs_spec, payoff in _candidates(relation, len(quotes)):
        legs: list[Leg] = []
        for index, side in legs_spec:
            quote = quotes[index]
            price = quote.ask(side)
            if price is None:
                result.missing_quotes.append(f'{quote.market_id}:{side}_ask')
                break
            fee = quote.fee_schedule().fee(price, size, maker=maker)
            legs.append(
                Leg(
                    market_id=quote.market_id,
                    venue=quote.venue,
                    side=side,
                    price=price,
                    contracts=contracts,
                    cost=_clean(price * contracts),
                    fee=float(fee),
                )
            )
        else:
            cost = _clean(sum(leg.cost for leg in legs))
            fees = _clean(sum(leg.fee for leg in legs))
            gross = _clean(payoff * contracts - cost)
            result.baskets.append(
                Basket(
                    description=description,
                    legs=legs,
                    payoff_per_contract=float(payoff),
                    contracts=contracts,
                    cost=cost,
                    fees=fees,
                    gross_edge=gross,
                    net_edge=_clean(gross - fees),
                )
            )
    return result
