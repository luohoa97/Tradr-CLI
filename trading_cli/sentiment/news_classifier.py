"""News event classifier — assigns importance weights to headlines by type.

Categorizes headlines into:
- earnings: earnings reports, guidance updates
- executive: CEO/CFO changes, board moves
- product: product launches, recalls, approvals
- macro: interest rates, CPI, unemployment, Fed policy
- generic: everything else (lower weight)

Each category has a configurable weight multiplier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(Enum):
    EARNINGS = "earnings"
    EXECUTIVE = "executive"
    PRODUCT = "product"
    MACRO = "macro"
    GENERIC = "generic"


@dataclass
class EventClassification:
    event_type: EventType
    weight: float
    confidence: float  # 0.0-1.0 how confident we are in the classification


# Default weights — higher means the headline is more impactful
DEFAULT_WEIGHTS: dict[EventType, float] = {
    EventType.EARNINGS: 1.5,    # earnings reports move markets significantly
    EventType.EXECUTIVE: 1.3,   # leadership changes signal strategic shifts
    EventType.PRODUCT: 1.2,     # product news affects company outlook
    EventType.MACRO: 1.4,       # macro news affects entire market
    EventType.GENERIC: 0.8,     # generic news has lower impact
}

# Keyword patterns for classification
EARNINGS_KEYWORDS = [
    r'\bearnings\b', r'\bprofit\b', r'\brevenue\b', r'\bloss\b',
    r'\bEPS\b', r'\bper share\b', r'\bquarterly\b.*\bresult',
    r'\bguidance\b', r'\bforecast\b', r'\boutlook\b',
    r'\bbeat.*expect', r'\bmiss.*expect', r'\banalyst.*expect',
    r'\breport.*earning', r'\bQ\d\b.*\bresult',
]

EXECUTIVE_KEYWORDS = [
    r'\bCEO\b', r'\bCFO\b', r'\bCOO\b', r'\bCTO\b',
    r'\bchief\s+(executive|financial|operating|technology)',
    r'\bresign', r'\bstep\s+down\b', r'\bappointed\b',
    r'\bnew\s+CEO\b', r'\bboard\b', r'\bdirector',
    r'\bleadership\b', r'\bexecutive\b',
]

PRODUCT_KEYWORDS = [
    r'\bproduct\s+launch', r'\brecall\b', r'\bFDA\b',
    r'\bapproval\b', r'\brecalled\b', r'\bnew\s+product',
    r'\biPhone\b', r'\biPad\b', r'\bTesla\b.*\bmodel',
    r'\bpipeline\b', r'\btrial\b', r'\bclinical\b',
    r'\bpatent\b', r'\binnovation\b',
]

MACRO_KEYWORDS = [
    r'\bFed\b', r'\bFederal\s+Reserve\b', r'\binterest\s+rate',
    r'\bCPI\b', r'\binflation\b', r'\bunemployment\b',
    r'\bjobs\s+report', r'\bGDP\b', r'\brecession\b',
    r'\btariff\b', r'\btrade\s+war\b', r'\bsanction',
    r'\bcentral\s+bank\b', r'\bmonetary\s+policy',
    r'\bquantitative\s+(easing|tightening)',
]


def classify_headline(headline: str, custom_weights: dict[EventType, float] | None = None) -> EventClassification:
    """Classify a headline into an event type and return its weight.

    Uses keyword matching with confidence based on how many keywords match.
    """
    text = headline.lower()
    weights = custom_weights or DEFAULT_WEIGHTS

    patterns = {
        EventType.EARNINGS: EARNINGS_KEYWORDS,
        EventType.EXECUTIVE: EXECUTIVE_KEYWORDS,
        EventType.PRODUCT: PRODUCT_KEYWORDS,
        EventType.MACRO: MACRO_KEYWORDS,
    }

    best_type = EventType.GENERIC
    best_confidence = 0.0

    for event_type, keyword_list in patterns.items():
        matches = sum(1 for kw in keyword_list if re.search(kw, text))
        if matches > 0:
            confidence = min(1.0, matches / 3.0)  # 3+ matches = high confidence
            if confidence > best_confidence:
                best_confidence = confidence
                best_type = event_type

    return EventClassification(
        event_type=best_type,
        weight=weights.get(best_type, 1.0),
        confidence=best_confidence,
    )


def classify_headlines(
    headlines: list[str],
    custom_weights: dict[EventType, float] | None = None,
) -> list[EventClassification]:
    """Classify multiple headlines at once."""
    return [classify_headline(h, custom_weights) for h in headlines]
