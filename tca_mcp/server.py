"""
TCA MCP Server — Topological reasoning as a native AI tool.

5 tools. One server. Any Claude instance gets structural analysis.
No API keys. No external calls. Pure topology.
"""

import json
import os
import re

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
import uvicorn

from api import engine

server = Server("tca")


# --- Text-to-Graph Parser (no AI, no API keys) ---

# Verbs/phrases that map to edge types.
_EDGE_PATTERNS = {
    "BOUNDS": [
        r"(?:controls?|constrains?|limits?|regulates?|governs?|manages?|oversees?|restricts?|bounds?)",
        r"(?:reports?\s+to|accountable\s+to|answers?\s+to|under)",
        r"(?:has\s+(?:authority|power|control)\s+over)",
    ],
    "EXPRESSES": [
        r"(?:produces?|creates?|generates?|builds?|makes?|delivers?|outputs?|drives?|causes?)",
        r"(?:leads?\s+to|results?\s+in|feeds?|provides?|supplies?|enables?)",
    ],
    "REMOVES": [
        r"(?:contradicts?|conflicts?\s+with|opposes?|undermines?|blocks?|prevents?)",
        r"(?:competes?(?:\s+(?:with|for|against))?|competing)",
        r"(?:fights?\s+(?:with|against|over|for)|clashes?\s+with|tensions?\s+with)",
        r"(?:at\s+odds\s+with|incompatible\s+with)",
    ],
    "SEEKS": [
        r"(?:seeks?|wants?|needs?|tries?\s+to|attempts?\s+to|aims?\s+to|strives?\s+for)",
        r"(?:lacks?|missing|no\s+(?:direct\s+)?(?:access|connection|link|feedback|influence))",
        r"(?:cannot|can(?:'|no)t|unable\s+to|has\s+no)",
    ],
    "INHERITS": [
        r"(?:depends?\s+on|relies?\s+on|derives?\s+from|based\s+on|inherits?|requires?)",
        r"(?:part\s+of|belongs?\s+to|subset\s+of|branch\s+of)",
    ],
    "MIRRORS": [
        r"(?:similar\s+to|parallels?|resembles?|like|mirrors?|analogous\s+to)",
        r"(?:same\s+as|equivalent\s+to|corresponds?\s+to)",
    ],
    "VERIFIES": [
        r"(?:proves?|confirms?|validates?|verifies?|demonstrates?|evidence\s+(?:for|that))",
        r"(?:shows?\s+that|ensures?|guarantees?)",
    ],
}

# Compile patterns.
_COMPILED_EDGE_PATTERNS = {}
for _etype, _pats in _EDGE_PATTERNS.items():
    _COMPILED_EDGE_PATTERNS[_etype] = re.compile(
        "|".join(_pats), re.IGNORECASE
    )


def _parse_text_to_graph(description: str) -> dict:
    """Parse plain English into a TCA graph. No AI. Pure regex + heuristics.

    Strategy:
    1. Split into sentences.
    2. Extract noun phrases from both sides of relationship verbs.
    3. Extract capitalized phrases, quoted terms, "X and Y" lists.
    4. Discover lowercase domain terms via frequency analysis.
    5. Detect relationship verbs to determine edge types.
    6. Build graph.
    """
    sentences = re.split(r'[.!?]+', description)
    sentences = [s.strip() for s in sentences if s.strip()]

    entity_mentions = {}  # nid -> label
    _merge_map = {}  # old_nid -> new_nid (for deduplication)

    def _to_id(label: str) -> str:
        return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')

    def _resolve_nid(nid: str) -> str:
        """Follow merge chain to find canonical nid."""
        while nid in _merge_map:
            nid = _merge_map[nid]
        return nid

    def _add_entity(label: str) -> str:
        label = label.strip()
        if not label or len(label) < 2:
            return ""
        # Strip leading articles.
        label = re.sub(r'^(?:the|a|an)\s+', '', label, flags=re.IGNORECASE).strip()
        if not label:
            return ""
        nid = _to_id(label)
        if not nid or len(nid) < 2:
            return ""
        # Check if already merged into something else.
        nid = _resolve_nid(nid)
        if nid in entity_mentions:
            return nid
        # Substring deduplication: check against existing entities.
        # If new nid contains an existing nid (or vice versa), merge.
        for existing_nid in list(entity_mentions.keys()):
            # Only merge if the shorter nid is at least 4 chars (avoid
            # overly generic matches like "roi" matching "battery_roi").
            if existing_nid in nid and len(existing_nid) >= 4:
                # New entity is a superset of existing → map to existing.
                _merge_map[nid] = existing_nid
                return existing_nid
            if nid in existing_nid and len(nid) >= 4 and '_' in nid:
                # Existing entity is a superset of new → absorb existing.
                # Only when new entity is multi-word (has underscore) to prevent
                # single words like "solar" absorbing "solar_production".
                entity_mentions.pop(existing_nid)
                entity_mentions[nid] = label
                _merge_map[existing_nid] = nid
                return nid
        entity_mentions[nid] = label
        return nid

    edges = []

    _STOP_WORDS = frozenset([
        'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
        'has', 'have', 'had', 'this', 'that', 'it', 'they', 'he', 'she',
        'all', 'every', 'each', 'any', 'some', 'no', 'not', 'if', 'then',
        'when', 'while', 'for', 'with', 'from', 'to', 'in', 'on', 'at',
        'by', 'who', 'which', 'what', 'how', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall',
        'its', 'their', 'his', 'her', 'our', 'your', 'my', 'of', 'about',
        'also', 'just', 'only', 'very', 'much', 'more', 'most', 'other',
        'so', 'too', 'as', 'than', 'such', 'both', 'either', 'neither',
        'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'once', 'here', 'there', 'where', 'why', 'how', 'been',
        'being', 'having', 'doing', 'those', 'these', 'same', 'own',
        'see', 'sees', 'seen', 'get', 'gets', 'got', 'become', 'becomes',
        'whether', 'because', 'since', 'unless', 'until', 'though',
    ])

    # Words that are never entities on their own.
    _JUNK_WORDS = frozenset([
        'problems', 'problem', 'issues', 'issue', 'things', 'thing',
        'way', 'ways', 'lot', 'lots', 'kind', 'type', 'part', 'parts',
        'direct', 'directly', 'new', 'old', 'big', 'small', 'good', 'bad',
        'not', 'both', 'simultaneously', 'highest', 'hours', 'priorities',
    ])

    # Common verbs to exclude from noun phrases.
    _VERBS = frozenset([
        'controls', 'control', 'manages', 'manage', 'builds', 'build',
        'sees', 'see', 'has', 'have', 'had', 'gets', 'get', 'does', 'do',
        'makes', 'make', 'takes', 'take', 'gives', 'give', 'keeps', 'keep',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'performs', 'perform', 'can', 'cannot', 'may', 'might',
        'produces', 'produce', 'creates', 'create', 'drives', 'drive',
        'generates', 'generate', 'leads', 'lead', 'causes', 'cause',
        'blocks', 'block', 'prevents', 'prevent', 'enables', 'enable',
        'seeks', 'seek', 'wants', 'want', 'needs', 'need',
        'proves', 'prove', 'confirms', 'confirm', 'shows', 'show',
        'competes', 'compete', 'opposes', 'oppose', 'contradicts', 'contradict',
        'reduces', 'reduce', 'increases', 'increase', 'offsets', 'offset',
        'depends', 'depend', 'relies', 'rely', 'requires', 'require',
        'mirrors', 'mirror', 'parallels', 'parallel', 'resembles', 'resemble',
        'regulates', 'regulate', 'governs', 'govern', 'restricts', 'restrict',
        'constrains', 'constrain', 'limits', 'limit', 'oversees', 'oversee',
        'reports', 'report', 'answers', 'answer', 'influences', 'influence',
        'affects', 'affect', 'supports', 'support', 'funds', 'fund',
        'billed', 'bill', 'using', 'used', 'use',
        'determines', 'determine', 'forces', 'force', 'destroys', 'destroy',
        'earns', 'earn', 'shortens', 'shorten', 'targets', 'target',
        'captures', 'capture', 'coincides', 'coincide', 'represents', 'represent',
        'varies', 'vary', 'discharges', 'discharge', 'covers', 'cover',
        'means', 'mean', 'allows', 'allow', 'runs', 'run', 'sets', 'set',
        'adds', 'add', 'pays', 'pay', 'saves', 'save', 'charges', 'charge',
        'finds', 'find', 'helps', 'help', 'works', 'work', 'starts', 'start',
        'turns', 'turn', 'puts', 'put', 'moves', 'move', 'calls', 'call',
        'goes', 'go', 'comes', 'come', 'knows', 'know', 'says', 'say',
        'tells', 'tell', 'holds', 'hold', 'stands', 'stand', 'sits', 'sit',
        'falls', 'fall', 'drops', 'drop', 'raises', 'raise', 'lowers', 'lower',
        'sends', 'send', 'brings', 'bring', 'leaves', 'leave',
        'divided', 'captured', 'billed', 'represented',
    ])

    # Auxiliary verbs — NEVER appear inside a noun phrase.
    _AUX_VERBS = frozenset([
        'is', 'are', 'was', 'were', 'has', 'have', 'had',
        'can', 'could', 'will', 'would', 'shall', 'should',
        'may', 'might', 'do', 'does', 'did',
    ])

    # Gerund/participle forms that are actually nouns (allowlist).
    _NOUN_GERUNDS = frozenset([
        'sizing', 'pricing', 'metering', 'shaving', 'cycling', 'billing',
        'building', 'rating', 'loading', 'cooling', 'heating', 'lighting',
        'roofing', 'wiring', 'piping', 'framing', 'grading', 'zoning',
        'parking', 'staffing', 'funding', 'branding', 'marketing',
        'engineering', 'manufacturing', 'computing', 'networking',
        'banking', 'trading', 'mining', 'logging', 'training', 'testing',
        'housing', 'shipping', 'processing', 'modeling', 'planning',
        'scheduling', 'dispatching', 'forecasting', 'monitoring',
        'storage', 'degradation', 'consumption',  # not -ing but included
    ])

    # Relationship verbs used to split sentences into subject/object.
    _REL_VERB_RE = re.compile(
        r'\b('
        # BOUNDS verbs
        r'controls?|constrains?|limits?|regulates?|governs?|manages?|oversees?|restricts?|bounds?'
        r'|reports?\s+to|accountable\s+to|answers?\s+to'
        r'|has\s+(?:authority|power|control)\s+over'
        # EXPRESSES verbs
        r'|produces?|creates?|generates?|builds?|makes?|delivers?|outputs?'
        r'|drives?|causes?|leads?\s+to|results?\s+in|feeds?|provides?|supplies?|enables?'
        r'|offsets?'
        # REMOVES verbs
        r'|contradicts?|conflicts?\s+with|opposes?|undermines?|blocks?|prevents?'
        r'|competes?\s+(?:with|for|against)|competing'
        r'|fights?\s+(?:with|against|over|for)|clashes?\s+with'
        # SEEKS verbs
        r'|seeks?|wants?|needs?|tries?\s+to|attempts?\s+to|aims?\s+to'
        r'|lacks?|cannot|can(?:\'|no)t|unable\s+to|has\s+no'
        # INHERITS verbs
        r'|depends?\s+on|relies?\s+on|derives?\s+from|based\s+on|inherits?|requires?'
        r'|part\s+of|belongs?\s+to'
        # MIRRORS verbs
        r'|similar\s+to|parallels?|resembles?|mirrors?|analogous\s+to'
        r'|same\s+as|equivalent\s+to|corresponds?\s+to'
        # VERIFIES verbs
        r'|proves?|confirms?|validates?|verifies?|demonstrates?'
        r'|shows?\s+that|ensures?|guarantees?'
        # Misc connectors
        r'|increases?\s+with'
        r')\b',
        re.IGNORECASE
    )

    def _is_leading_gerund(word: str) -> bool:
        """Check if a word is a verb gerund (not a noun-gerund)."""
        w = word.lower()
        return (w.endswith('ing') or w.endswith('ed')) and w not in _NOUN_GERUNDS

    def _clean_noun_phrase(phrase: str) -> str:
        """Clean a noun phrase: strip function words from edges, limit length."""
        phrase = phrase.strip(' ,;:')
        # Strip leading words: stop words, junk words, verbs, AND verb gerunds.
        # Strip trailing words: stop words, junk words, trailing adjectives
        # (not verbs — "support", "control" etc. are often nouns at end).
        changed = True
        while changed:
            changed = False
            words = phrase.split()
            if not words:
                break
            w0 = words[0].lower()
            if w0 in _STOP_WORDS | _JUNK_WORDS | _VERBS or _is_leading_gerund(w0):
                phrase = ' '.join(words[1:])
                changed = True
            words = phrase.split()
            if words:
                tail = words[-1].lower()
                if tail in _STOP_WORDS | _JUNK_WORDS | _TRAILING_ADJ:
                    phrase = ' '.join(words[:-1])
                    changed = True
                # Also strip trailing conjugated verbs (-s/-ed forms).
                elif tail in _VERBS and (tail.endswith('s') or tail.endswith('ed')):
                    phrase = ' '.join(words[:-1])
                    changed = True
        # Limit to max 4 words.
        words = phrase.split()
        if len(words) > 4:
            phrase = ' '.join(words[-4:])
        return phrase.strip()

    def _split_clause(text: str) -> list[str]:
        """Split a clause into sub-phrases on conjunctions, prepositions, punctuation."""
        parts = re.split(
            r'\s*(?:,|;)\s*'
            r'|\s+(?:and|or|but|with|who|which|that|while|although|however|yet'
            r'|for|during|on|at|in|by|from|into|through|about|against|of|to)\s+',
            text, flags=re.IGNORECASE
        )
        return [p.strip() for p in parts if p.strip()]

    def _extract_noun_phrase(text: str) -> list[str]:
        """Extract noun phrases from a clause. Handles lowercase compound nouns."""
        text = text.strip()
        if not text:
            return []

        results = []

        # 1. Quoted terms: "demand shaving", 'energy arbitrage'
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        for q in quoted:
            q = _clean_noun_phrase(q)
            if q:
                results.append(q)

        # 2. Parenthetical terms: (time-of-use pricing)
        parens = re.findall(r'\(([^)]+)\)', text)
        for p in parens:
            p = _clean_noun_phrase(p)
            if p:
                results.append(p)

        # Remove quoted/parenthetical from text so we don't double-extract.
        clean = re.sub(r'["\'][^"\']+["\']', ' ', text)
        clean = re.sub(r'\([^)]+\)', ' ', clean)

        # 3. Split clause into sub-phrases, clean each.
        parts = _split_clause(clean)
        for part in parts:
            part = _clean_noun_phrase(part)
            if part and _is_valid_entity(part):
                results.append(part)

        # 4. Capitalized phrases with optional lowercase continuation:
        #    "Customer Support", "Load profile", "CEO"
        cap_extended = re.findall(
            r'\b([A-Z][a-zA-Z]*(?:\s+[a-zA-Z][a-z]+)*)\b', clean
        )
        for c in cap_extended:
            c = _clean_noun_phrase(c)
            if not c or not _is_valid_entity(c) or len(c) <= 2:
                continue
            # Skip if internal words contain verbs/auxiliaries (greedy capture).
            c_words = c.lower().split()
            if len(c_words) > 1 and any(w in _AUX_VERBS | _VERBS for w in c_words[1:]):
                continue
            results.append(c)

        # 5. Known domain terms (from frequency analysis) in this clause.
        clause_lower = clean.lower()
        for term in domain_terms:
            if term in clause_lower:
                results.append(term)

        return results

    # Adjectives that shouldn't end a noun phrase.
    _TRAILING_ADJ = frozenset([
        'catastrophic', 'aggressive', 'missed', 'highest', 'lowest',
        'commercial', 'direct', 'indirect', 'critical', 'major', 'minor',
        'significant', 'total', 'partial', 'full', 'empty', 'open', 'closed',
        'profitable', 'optimal', 'conservative', 'annual', 'single',
        'uncertain', 'simultaneous', 'available', 'typical', 'maximum',
        'minimum', 'average', 'potential', 'actual', 'expected',
    ])

    def _is_valid_entity(phrase: str) -> bool:
        """Check if a phrase is a real entity, not a verb or junk."""
        words = phrase.lower().split()
        if not words:
            return False
        # All stop words = not an entity.
        if all(w in _STOP_WORDS or w in _JUNK_WORDS for w in words):
            return False
        # Single character or too short.
        if len(phrase) < 3:
            return False
        # Contains conjunction in the middle → not a clean noun phrase.
        if any(w in ('and', 'or', 'but') for w in words):
            return False
        # Starts with a common verb (false positive).
        if words[0] in _VERBS:
            return False
        # Starts with a gerund/participle that isn't a known noun-gerund.
        if (words[0].endswith('ing') or words[0].endswith('ed')) \
                and words[0] not in _NOUN_GERUNDS:
            return False
        # Contains an auxiliary verb internally → it's a clause, not a noun phrase.
        # "demand charges can represent" → has "can" at position 2
        if len(words) > 1:
            for w in words[1:]:
                if w in _AUX_VERBS:
                    return False
        # Internal verb (positions 1 to n-2) → it's a clause.
        # "arbitrage shortens battery life" → "shortens" at position 1
        if len(words) >= 3:
            for w in words[1:-1]:
                if w in _VERBS:
                    return False
        # Trailing verb detection.
        # For 3+ word phrases: reject if trailing word is in _VERBS (any form).
        #   "load peaks coincide" → "coincide" in _VERBS → reject.
        # For 2-word phrases: only reject conjugated forms (-s/-ed) to preserve
        #   noun uses like "Customer Support" where "support" is in _VERBS.
        if len(words) > 1 and words[-1] in _VERBS:
            if len(words) >= 3 or words[-1].endswith('s') or words[-1].endswith('ed'):
                return False
        # Ends with a bare adjective (no noun after it).
        if words[-1] in _TRAILING_ADJ:
            return False
        # Pure numbers.
        if re.match(r'^[\d\s.,-]+$', phrase):
            return False
        # Single generic words that aren't useful entities alone.
        if len(words) == 1 and words[0] in (
            'load', 'generation', 'rate', 'demand', 'energy', 'net',
            'company', 'organization', 'system', 'structure', 'process',
            'solar', 'battery', 'building', 'season', 'peaks', 'peak',
            'cost', 'value', 'size', 'capacity', 'power', 'output',
            'input', 'goal', 'plan', 'budget', 'revenue', 'savings',
            'model', 'data', 'level', 'interval', 'period', 'percent',
            'result', 'impact', 'effect', 'factor', 'rule', 'clause',
        ):
            return False
        return True

    # --- Phase 1: Frequency-based domain term discovery ---
    # Count 2-word and 3-word lowercase phrases across all sentences.
    # Phrases appearing 2+ times are likely domain terms.
    bigram_counts: dict[str, int] = {}
    full_text_lower = description.lower()
    # Extract all 2-3 word sequences.
    words_all = re.findall(r'[a-z][\w-]*', full_text_lower)
    for n in (2, 3):
        for i in range(len(words_all) - n + 1):
            gram_words = words_all[i:i+n]
            # Skip if first or last word is a stop word/verb/junk.
            if gram_words[0] in _STOP_WORDS | _JUNK_WORDS | _VERBS:
                continue
            if gram_words[-1] in _STOP_WORDS | _JUNK_WORDS | _VERBS:
                continue
            gram = ' '.join(gram_words)
            bigram_counts[gram] = bigram_counts.get(gram, 0) + 1

    # Domain terms: multi-word phrases appearing 2+ times.
    domain_terms = set()
    for gram, count in bigram_counts.items():
        if count >= 2:
            domain_terms.add(gram)

    # Pre-register domain terms as entities.
    for term in domain_terms:
        if _is_valid_entity(term):
            _add_entity(term)

    # --- Phase 2: Sentence-by-sentence entity and edge extraction ---
    for sentence in sentences:
        # Split sentence at relationship verb to get subject and object clauses.
        verb_match = _REL_VERB_RE.search(sentence)

        found_entities = []

        if verb_match:
            subject_clause = sentence[:verb_match.start()]
            verb_text = verb_match.group(0)
            object_clause = sentence[verb_match.end():]

            # Extract entities from subject side.
            subj_phrases = _extract_noun_phrase(subject_clause)
            for phrase in subj_phrases:
                nid = _add_entity(phrase)
                if nid:
                    found_entities.append(nid)

            # Extract entities from object side.
            obj_phrases = _extract_noun_phrase(object_clause)
            for phrase in obj_phrases:
                nid = _add_entity(phrase)
                if nid:
                    found_entities.append(nid)

            # Also check for a second verb match in the object clause
            # e.g. "Customer Support sees problems but cannot influence Engineering priorities"
            verb_match2 = _REL_VERB_RE.search(object_clause)
            if verb_match2:
                obj2_clause = object_clause[verb_match2.end():]
                obj2_phrases = _extract_noun_phrase(obj2_clause)
                for phrase in obj2_phrases:
                    nid = _add_entity(phrase)
                    if nid:
                        found_entities.append(nid)
        else:
            # No relationship verb found — extract all noun phrases.
            all_phrases = _extract_noun_phrase(sentence)
            for phrase in all_phrases:
                nid = _add_entity(phrase)
                if nid:
                    found_entities.append(nid)

        # Also scan for domain terms explicitly in this sentence.
        sent_lower = sentence.lower()
        for term in domain_terms:
            if term in sent_lower and _is_valid_entity(term):
                nid = _add_entity(term)
                if nid and nid not in found_entities:
                    found_entities.append(nid)

        # Also extract capitalized phrases that might have been missed.
        # Skip single capitalized words at sentence start (just capitalization convention).
        caps = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', sentence)
        for ci, c in enumerate(caps):
            c = _clean_noun_phrase(c)
            if not c or not _is_valid_entity(c):
                continue
            # Single word at sentence start is likely just capitalization, not an entity.
            # Keep it only if it's an acronym (all caps) or appears capitalized mid-sentence too.
            if ' ' not in c and ci == 0 and not c.isupper():
                # Check if this word appears capitalized elsewhere (mid-sentence).
                word_re = re.compile(r'(?<!^)(?<![.!?]\s)\b' + re.escape(c) + r'\b')
                if not word_re.search(description):
                    continue
            nid = _add_entity(c)
            if nid and nid not in found_entities:
                found_entities.append(nid)

        # Deduplicate while preserving order.
        seen = set()
        unique_entities = []
        for e in found_entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)
        found_entities = unique_entities

        if len(found_entities) < 2:
            continue

        # Detect edge type from sentence.
        edge_type = "EXPRESSES"  # default
        for etype, pattern in _COMPILED_EDGE_PATTERNS.items():
            if pattern.search(sentence):
                edge_type = etype
                break

        # For "reports to" / "accountable to" — direction is reversed
        # (A reports to B means B BOUNDS A).
        reverse = bool(re.search(
            r'(?:reports?\s+to|accountable\s+to|answers?\s+to|under\b)',
            sentence, re.IGNORECASE
        ))

        # Create edges between consecutive entity pairs in the sentence.
        for i in range(len(found_entities) - 1):
            src, tgt = found_entities[i], found_entities[i + 1]
            if src == tgt:
                continue
            if reverse and edge_type == "BOUNDS":
                src, tgt = tgt, src
            edges.append({
                "source": src,
                "target": tgt,
                "type": edge_type,
                "weight": 1.0,
            })

    # Remap edges through merge map.
    for edge in edges:
        edge["source"] = _resolve_nid(edge["source"])
        edge["target"] = _resolve_nid(edge["target"])

    # Build node list.
    nodes = [{"id": nid, "label": label} for nid, label in entity_mentions.items()]

    # Deduplicate edges.
    seen_edges = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    # Derive a name from the first few entities.
    name = "System"
    if nodes:
        name = " / ".join(n["label"] for n in nodes[:3])

    return {
        "name": name,
        "description": description,
        "nodes": nodes,
        "edges": unique_edges,
    }


# --- Tool Definitions ---

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="tca_analyze_text",
            description=(
                "TCA — Topological Cognitive Architecture. A deterministic structural analysis engine. "
                "Zero AI. Zero learned parameters. You give it a graph. It does math on the shape of "
                "connections and returns what's broken. Think of it as a calculator for system structure.\n\n"

                "YOU ARE THE KEYBOARD. TCA IS THE CALCULATOR. THE USER IS ASKING THE QUESTION.\n"
                "You do NOT analyze systems yourself when TCA is available. You BUILD the graph and "
                "let TCA analyze it. Your job is translating the user's question into a topological "
                "graph. TCA's job is finding what's structurally broken. Never mix these roles.\n\n"

                "WHEN TO USE: Any time the user asks about the STRUCTURE of something — 'What's broken "
                "about X?', 'Analyze the structure of...', 'What are the contradictions in...', "
                "'Why does X keep failing?', any system, company, country, religion, plan, relationship, "
                "ideology, business model, technology stack. Also for comparing two things structurally "
                "or checking a plan for hidden dependencies.\n"
                "When NOT to use: simple facts, coding, math, creative writing.\n\n"

                "HOW TO BUILD A GRAPH:\n"
                "Pass the 'description' parameter as a JSON string with this format:\n"
                '{"name": "System Name", "nodes": [{"id": "snake_case", "label": "Human Label"}], '
                '"edges": [{"source": "from_id", "target": "to_id", "type": "EDGE_TYPE", "weight": 1.0}]}\n\n'

                "THE 7 EDGE TYPES (this is the core — choosing the right type is EVERYTHING):\n"
                "MIRRORS — A reflects/parallels B. Structural analogy.\n"
                "INHERITS — A derives from/depends on B. A couldn't exist without B.\n"
                "BOUNDS — A constrains/limits/controls B. Power structure.\n"
                "EXPRESSES — A produces/causes/creates B. Direct causal output.\n"
                "VERIFIES — A proves/grounds B with evidence. A provides evidence B is real.\n"
                "REMOVES — A contradicts/destroys/undermines B. Structural conflict.\n"
                "SEEKS — A wants B but hasn't proven it. Aspiration. Unverified claim.\n\n"

                "CRITICAL RULES:\n"
                "1. If it's unproven, use SEEKS not VERIFIES. Most plans are mostly SEEKS edges.\n"
                "2. ALWAYS include REMOVES edges. Every real system has contradictions. Zero REMOVES = naive.\n"
                "3. ALWAYS include BOUNDS edges. Every system has power structures and constraints.\n"
                "4. Don't make everything EXPRESSES. Ask: Is this actually BOUNDS? REMOVES? SEEKS? INHERITS?\n"
                "5. 10-25 nodes. Under 10 is too simple. Over 25 is noise.\n"
                "6. Model the tensions, not just the entities. Include the shadow structure — informal "
                "power, unspoken conflicts, aspirational claims that aren't proven.\n\n"

                "HOW TO READ OUTPUT:\n"
                "- confidence: 0.0-1.0. Under 0.4 = structurally fragile.\n"
                "- problems: contradictions (REMOVES edges — present these FIRST, most valuable), "
                "feedback_traps (circular loops), dead_ends (sinks with no output), "
                "star_topologies (single points of failure).\n"
                "- questions: SEEKS edges — unproven assumptions.\n"
                "- solutions: proposed structural fixes ranked by confidence.\n\n"

                "PRESENTING RESULTS: Lead with contradictions. Then feedback traps. Then dead ends. "
                "Then unproven assumptions. Then solutions. Use the user's language — say 'Revenue "
                "contradicts your open source goal' NOT 'REMOVES edge detected'. State the confidence.\n\n"

                "WORKFLOW: User describes system → You build 10-25 node graph with honest edge types → "
                "Call tca_analyze_text with JSON graph → Present findings → User says 'fix #2' → "
                "Call tca_apply → Present what changed → User wants to save → Call tca_export.\n\n"

                "COMMON MISTAKES:\n"
                "- All EXPRESSES edges (didn't think about power, conflict, or uncertainty)\n"
                "- Zero REMOVES edges (hiding contradictions)\n"
                "- Under 7 nodes (not enough structure for patterns)\n"
                "- Using VERIFIES for hopes (if not proven, it's SEEKS)\n"
                "- Analyzing the system yourself instead of letting TCA do it\n"
                "- Presenting raw graph terminology to users\n\n"

                "Also accepts plain English as fallback — TCA will attempt to parse entities and "
                "relationships automatically, but structured JSON graphs produce far better results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "PREFERRED: A JSON string with format "
                            '{"name": "...", "nodes": [{"id": "...", "label": "..."}], '
                            '"edges": [{"source": "...", "target": "...", "type": "EDGE_TYPE", "weight": 1.0}]}. '
                            "Edge types: MIRRORS, INHERITS, BOUNDS, EXPRESSES, VERIFIES, REMOVES, SEEKS. "
                            "FALLBACK: Plain English description of the system (less accurate)."
                        ),
                    },
                },
                "required": ["description"],
            },
        ),
        Tool(
            name="tca_template",
            description=(
                "Load a pre-built TCA graph with instant analysis. "
                "Templates: economics, tanakh, us_geopolitics, "
                "china_geopolitics, apple, openai."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "template_name": {
                        "type": "string",
                        "enum": [
                            "economics", "tanakh", "us_geopolitics",
                            "china_geopolitics", "apple", "openai",
                        ],
                    }
                },
                "required": ["template_name"],
            },
        ),
        Tool(
            name="tca_solve",
            description=(
                "Get TCA solutions for an existing graph. Returns proposed "
                "structural fixes sorted by confidence."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"}
                },
                "required": ["graph_id"],
            },
        ),
        Tool(
            name="tca_apply",
            description=(
                "Apply a TCA solution to a graph by index. Modifies the "
                "graph and returns updated analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"},
                    "solution_index": {"type": "integer"},
                },
                "required": ["graph_id", "solution_index"],
            },
        ),
        Tool(
            name="tca_export",
            description=(
                "Export a TCA graph. Format 'json' = full state, "
                "'boot' = architecture only (no private data, safe for "
                "open source)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["json", "boot"],
                    },
                },
                "required": ["graph_id", "format"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        result = _handle_tool(name, arguments)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)}),
        )]


def _handle_tool(name: str, arguments: dict) -> dict:
    if name == "tca_analyze_text":
        if "description" not in arguments:
            return {"error": "Missing 'description' parameter. Pass a plain text description of the system to analyze."}
        return _analyze_text(arguments["description"])
    elif name == "tca_template":
        return _load_template(arguments["template_name"])
    elif name == "tca_solve":
        return _solve(arguments["graph_id"])
    elif name == "tca_apply":
        return _apply(arguments["graph_id"], arguments["solution_index"])
    elif name == "tca_export":
        return _export(arguments["graph_id"], arguments["format"])
    else:
        return {"error": f"Unknown tool: {name}"}


def _analyze_text(description: str) -> dict:
    """Text -> Graph -> Analysis.

    Accepts EITHER:
      1. A JSON graph object (string): {"name": ..., "nodes": [...], "edges": [...]}
         — Claude builds the graph, TCA does the math. Primary workflow.
      2. Plain English description of a system.
         — Text parser extracts entities and relationships. Fallback.
    """
    # Check if the description is actually a JSON graph object.
    try:
        parsed = json.loads(description)
        if isinstance(parsed, dict) and "nodes" in parsed and "edges" in parsed:
            return _analyze_graph(parsed)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass  # Not JSON — fall through to text parser.

    graph_data = _parse_text_to_graph(description)

    if not graph_data.get("nodes"):
        return {
            "error": (
                "Could not extract entities from the description. "
                "Try naming specific entities with capital letters: "
                "'The CEO controls Marketing. Sales competes with Marketing for budget.'"
            )
        }

    return _analyze_graph(graph_data)


def _analyze_graph(graph_data: dict) -> dict:
    """Graph -> Analysis. Loads graph data into TCA engine and runs analysis."""
    result = engine.create_graph(name=graph_data.get("name", "Auto"))
    graph_id = result["graph_id"]

    for node in graph_data.get("nodes", []):
        engine.add_node(graph_id, node["label"], node["id"])

    for edge in graph_data.get("edges", []):
        try:
            engine.add_edge(
                graph_id,
                edge["source"],
                edge["target"],
                edge["type"],
                edge.get("weight", 1.0),
            )
        except (ValueError, KeyError):
            continue

    analysis = engine.run_analysis(graph_id)

    return {
        "graph_id": graph_id,
        "graph": graph_data,
        "analysis": analysis,
    }


def _load_template(template_name: str) -> dict:
    result = engine.create_graph(
        name=template_name.replace("_", " ").title(),
        template=template_name,
    )
    graph_id = result["graph_id"]
    analysis = engine.run_analysis(graph_id)
    return {
        "graph_id": graph_id,
        "template": template_name,
        "analysis": analysis,
    }


def _solve(graph_id: str) -> dict:
    analysis = engine.run_analysis(graph_id)
    if analysis is None:
        return {"error": f"Graph '{graph_id}' not found"}
    solutions = analysis.get("solutions", [])
    return {"solutions": solutions, "total": len(solutions)}


def _apply(graph_id: str, index: int) -> dict:
    result = engine.apply_solution(graph_id, index)
    if result is None:
        return {"error": f"Graph '{graph_id}' not found"}
    return result


def _export(graph_id: str, fmt: str) -> dict:
    if fmt == "boot":
        return engine.export_boot(graph_id) or {"error": "Graph not found"}
    return engine.export_state(graph_id) or {"error": "Graph not found"}


# --- SSE Transport ---

from fastapi import FastAPI, Request
from starlette.responses import Response
from starlette.routing import Route

sse = SseServerTransport("/messages/")


async def handle_sse_endpoint(request: Request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )
    return Response()


async def handle_messages_endpoint(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)


routes = [
    Route("/sse", handle_sse_endpoint),
    Route("/messages/", handle_messages_endpoint, methods=["POST"]),
]

app = FastAPI(routes=routes)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8100))
    uvicorn.run(app, host="0.0.0.0", port=port)
