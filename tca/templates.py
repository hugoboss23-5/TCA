"""
TCA Templates — Pre-built graphs for instant analysis.

6 templates: economics, tanakh, us_geopolitics, china_geopolitics, apple, openai.
Each models a real system with honest edge types (including contradictions).
"""

from __future__ import annotations

from tca.graph import TopologicalGraph, EdgeType

AVAILABLE = [
    "economics", "tanakh", "us_geopolitics",
    "china_geopolitics", "apple", "openai",
]


def apply_template(g: TopologicalGraph, template_name: str) -> None:
    """Load a pre-built template into a graph."""
    builders = {
        "economics": _economics,
        "tanakh": _tanakh,
        "us_geopolitics": _us_geopolitics,
        "china_geopolitics": _china_geopolitics,
        "apple": _apple,
        "openai": _openai,
    }
    builder = builders.get(template_name)
    if builder is None:
        raise ValueError(f"Unknown template: {template_name}. "
                         f"Available: {AVAILABLE}")
    builder(g)


def _economics(g: TopologicalGraph) -> None:
    g.add_node(label="Central Banks", node_id="central_banks")
    g.add_node(label="Interest Rates", node_id="interest_rates")
    g.add_node(label="Inflation", node_id="inflation")
    g.add_node(label="Employment", node_id="employment")
    g.add_node(label="GDP Growth", node_id="gdp_growth")
    g.add_node(label="Consumer Spending", node_id="consumer_spending")
    g.add_node(label="Government Debt", node_id="gov_debt")
    g.add_node(label="Trade Balance", node_id="trade_balance")
    g.add_node(label="Currency Value", node_id="currency")
    g.add_node(label="Stock Market", node_id="stock_market")
    g.add_node(label="Housing Market", node_id="housing")
    g.add_node(label="Wealth Inequality", node_id="inequality")
    g.add_node(label="Technological Innovation", node_id="tech_innovation")
    g.add_node(label="Supply Chains", node_id="supply_chains")
    g.add_node(label="Energy Prices", node_id="energy")

    g.add_edge("central_banks", "interest_rates", EdgeType.BOUNDS, 0.9)
    g.add_edge("interest_rates", "inflation", EdgeType.REMOVES, 0.8)
    g.add_edge("interest_rates", "housing", EdgeType.BOUNDS, 0.7)
    g.add_edge("inflation", "consumer_spending", EdgeType.REMOVES, 0.6)
    g.add_edge("inflation", "currency", EdgeType.REMOVES, 0.5)
    g.add_edge("employment", "consumer_spending", EdgeType.EXPRESSES, 0.8)
    g.add_edge("consumer_spending", "gdp_growth", EdgeType.EXPRESSES, 0.9)
    g.add_edge("gdp_growth", "employment", EdgeType.EXPRESSES, 0.7)
    g.add_edge("gdp_growth", "stock_market", EdgeType.MIRRORS, 0.6)
    g.add_edge("gov_debt", "interest_rates", EdgeType.SEEKS, 0.5)
    g.add_edge("gov_debt", "gdp_growth", EdgeType.REMOVES, 0.4)
    g.add_edge("trade_balance", "currency", EdgeType.EXPRESSES, 0.6)
    g.add_edge("trade_balance", "gdp_growth", EdgeType.EXPRESSES, 0.5)
    g.add_edge("currency", "trade_balance", EdgeType.BOUNDS, 0.5)
    g.add_edge("stock_market", "consumer_spending", EdgeType.EXPRESSES, 0.4)
    g.add_edge("housing", "consumer_spending", EdgeType.EXPRESSES, 0.5)
    g.add_edge("inequality", "consumer_spending", EdgeType.REMOVES, 0.6)
    g.add_edge("inequality", "gdp_growth", EdgeType.SEEKS, 0.3)
    g.add_edge("tech_innovation", "gdp_growth", EdgeType.EXPRESSES, 0.7)
    g.add_edge("tech_innovation", "employment", EdgeType.REMOVES, 0.4)
    g.add_edge("supply_chains", "inflation", EdgeType.EXPRESSES, 0.6)
    g.add_edge("energy", "inflation", EdgeType.EXPRESSES, 0.7)
    g.add_edge("energy", "supply_chains", EdgeType.BOUNDS, 0.5)


def _tanakh(g: TopologicalGraph) -> None:
    g.add_node(label="HaShem (The Name)", node_id="hashem")
    g.add_node(label="Torah (Law)", node_id="torah")
    g.add_node(label="Covenant", node_id="covenant")
    g.add_node(label="Israel (People)", node_id="israel")
    g.add_node(label="Temple", node_id="temple")
    g.add_node(label="Prophets", node_id="prophets")
    g.add_node(label="Exile", node_id="exile")
    g.add_node(label="Return", node_id="return")
    g.add_node(label="Justice (Mishpat)", node_id="justice")
    g.add_node(label="Mercy (Chesed)", node_id="mercy")
    g.add_node(label="Idolatry", node_id="idolatry")
    g.add_node(label="Repentance (Teshuvah)", node_id="teshuvah")
    g.add_node(label="Wisdom (Chokmah)", node_id="wisdom")
    g.add_node(label="Creation", node_id="creation")

    g.add_edge("hashem", "torah", EdgeType.EXPRESSES, 1.0)
    g.add_edge("hashem", "covenant", EdgeType.BOUNDS, 0.9)
    g.add_edge("hashem", "creation", EdgeType.EXPRESSES, 1.0)
    g.add_edge("torah", "israel", EdgeType.BOUNDS, 0.9)
    g.add_edge("covenant", "israel", EdgeType.BOUNDS, 0.8)
    g.add_edge("israel", "temple", EdgeType.EXPRESSES, 0.7)
    g.add_edge("prophets", "israel", EdgeType.VERIFIES, 0.8)
    g.add_edge("prophets", "justice", EdgeType.EXPRESSES, 0.9)
    g.add_edge("idolatry", "covenant", EdgeType.REMOVES, 0.9)
    g.add_edge("idolatry", "exile", EdgeType.EXPRESSES, 0.8)
    g.add_edge("exile", "return", EdgeType.SEEKS, 0.7)
    g.add_edge("teshuvah", "return", EdgeType.EXPRESSES, 0.8)
    g.add_edge("teshuvah", "covenant", EdgeType.VERIFIES, 0.7)
    g.add_edge("justice", "mercy", EdgeType.MIRRORS, 0.6)
    g.add_edge("justice", "torah", EdgeType.INHERITS, 0.8)
    g.add_edge("mercy", "hashem", EdgeType.INHERITS, 0.9)
    g.add_edge("wisdom", "torah", EdgeType.MIRRORS, 0.7)
    g.add_edge("wisdom", "creation", EdgeType.VERIFIES, 0.6)


def _us_geopolitics(g: TopologicalGraph) -> None:
    g.add_node(label="US Military", node_id="us_military")
    g.add_node(label="US Dollar (Reserve Currency)", node_id="usd")
    g.add_node(label="Tech Giants", node_id="tech_giants")
    g.add_node(label="NATO Alliance", node_id="nato")
    g.add_node(label="China (Rival)", node_id="china")
    g.add_node(label="Russia (Adversary)", node_id="russia")
    g.add_node(label="Middle East Oil", node_id="mideast_oil")
    g.add_node(label="Domestic Polarization", node_id="polarization")
    g.add_node(label="Immigration", node_id="immigration")
    g.add_node(label="National Debt", node_id="national_debt")
    g.add_node(label="AI Supremacy Race", node_id="ai_race")
    g.add_node(label="Semiconductor Supply", node_id="semiconductors")
    g.add_node(label="Energy Independence", node_id="energy_indep")

    g.add_edge("us_military", "nato", EdgeType.EXPRESSES, 0.9)
    g.add_edge("us_military", "usd", EdgeType.VERIFIES, 0.7)
    g.add_edge("usd", "national_debt", EdgeType.SEEKS, 0.6)
    g.add_edge("tech_giants", "ai_race", EdgeType.EXPRESSES, 0.8)
    g.add_edge("tech_giants", "semiconductors", EdgeType.SEEKS, 0.7)
    g.add_edge("china", "us_military", EdgeType.REMOVES, 0.6)
    g.add_edge("china", "semiconductors", EdgeType.SEEKS, 0.8)
    g.add_edge("china", "ai_race", EdgeType.REMOVES, 0.7)
    g.add_edge("russia", "nato", EdgeType.REMOVES, 0.8)
    g.add_edge("russia", "energy_indep", EdgeType.REMOVES, 0.5)
    g.add_edge("mideast_oil", "energy_indep", EdgeType.BOUNDS, 0.6)
    g.add_edge("polarization", "nato", EdgeType.REMOVES, 0.4)
    g.add_edge("polarization", "immigration", EdgeType.MIRRORS, 0.5)
    g.add_edge("national_debt", "us_military", EdgeType.BOUNDS, 0.5)
    g.add_edge("semiconductors", "ai_race", EdgeType.BOUNDS, 0.8)
    g.add_edge("energy_indep", "usd", EdgeType.VERIFIES, 0.5)


def _china_geopolitics(g: TopologicalGraph) -> None:
    g.add_node(label="CCP (Party)", node_id="ccp")
    g.add_node(label="Belt and Road", node_id="bri")
    g.add_node(label="PLA (Military)", node_id="pla")
    g.add_node(label="Manufacturing Base", node_id="manufacturing")
    g.add_node(label="Taiwan Question", node_id="taiwan")
    g.add_node(label="Demographic Decline", node_id="demographics")
    g.add_node(label="Tech Self-Sufficiency", node_id="tech_self")
    g.add_node(label="Yuan Internationalization", node_id="yuan")
    g.add_node(label="Social Control (Credit)", node_id="social_credit")
    g.add_node(label="Real Estate Crisis", node_id="real_estate")
    g.add_node(label="US Containment", node_id="us_containment")
    g.add_node(label="Resource Security", node_id="resources")

    g.add_edge("ccp", "pla", EdgeType.BOUNDS, 0.9)
    g.add_edge("ccp", "social_credit", EdgeType.EXPRESSES, 0.8)
    g.add_edge("ccp", "bri", EdgeType.EXPRESSES, 0.7)
    g.add_edge("manufacturing", "bri", EdgeType.EXPRESSES, 0.6)
    g.add_edge("manufacturing", "demographics", EdgeType.SEEKS, 0.7)
    g.add_edge("taiwan", "pla", EdgeType.SEEKS, 0.8)
    g.add_edge("taiwan", "tech_self", EdgeType.BOUNDS, 0.9)
    g.add_edge("demographics", "manufacturing", EdgeType.REMOVES, 0.7)
    g.add_edge("demographics", "real_estate", EdgeType.EXPRESSES, 0.6)
    g.add_edge("tech_self", "us_containment", EdgeType.REMOVES, 0.7)
    g.add_edge("us_containment", "taiwan", EdgeType.VERIFIES, 0.6)
    g.add_edge("us_containment", "tech_self", EdgeType.BOUNDS, 0.8)
    g.add_edge("yuan", "bri", EdgeType.INHERITS, 0.5)
    g.add_edge("yuan", "us_containment", EdgeType.SEEKS, 0.4)
    g.add_edge("real_estate", "ccp", EdgeType.SEEKS, 0.6)
    g.add_edge("resources", "bri", EdgeType.INHERITS, 0.6)
    g.add_edge("resources", "manufacturing", EdgeType.BOUNDS, 0.5)


def _apple(g: TopologicalGraph) -> None:
    g.add_node(label="Hardware Design", node_id="hardware")
    g.add_node(label="Software Ecosystem", node_id="software")
    g.add_node(label="Services Revenue", node_id="services")
    g.add_node(label="Supply Chain (Foxconn)", node_id="supply_chain")
    g.add_node(label="Brand Premium", node_id="brand")
    g.add_node(label="Privacy Positioning", node_id="privacy")
    g.add_node(label="Developer Platform", node_id="developers")
    g.add_node(label="Chip Design (M-series)", node_id="chips")
    g.add_node(label="China Market Dependency", node_id="china_market")
    g.add_node(label="AI Integration", node_id="ai")
    g.add_node(label="Regulatory Pressure", node_id="regulation")
    g.add_node(label="Walled Garden", node_id="walled_garden")

    g.add_edge("hardware", "software", EdgeType.MIRRORS, 0.9)
    g.add_edge("hardware", "chips", EdgeType.INHERITS, 0.8)
    g.add_edge("software", "developers", EdgeType.EXPRESSES, 0.8)
    g.add_edge("software", "services", EdgeType.EXPRESSES, 0.7)
    g.add_edge("services", "brand", EdgeType.VERIFIES, 0.6)
    g.add_edge("brand", "hardware", EdgeType.EXPRESSES, 0.7)
    g.add_edge("privacy", "brand", EdgeType.VERIFIES, 0.8)
    g.add_edge("privacy", "ai", EdgeType.REMOVES, 0.5)
    g.add_edge("walled_garden", "developers", EdgeType.BOUNDS, 0.8)
    g.add_edge("walled_garden", "services", EdgeType.EXPRESSES, 0.7)
    g.add_edge("walled_garden", "regulation", EdgeType.SEEKS, 0.6)
    g.add_edge("regulation", "walled_garden", EdgeType.REMOVES, 0.7)
    g.add_edge("supply_chain", "china_market", EdgeType.MIRRORS, 0.6)
    g.add_edge("supply_chain", "hardware", EdgeType.BOUNDS, 0.7)
    g.add_edge("china_market", "services", EdgeType.BOUNDS, 0.5)
    g.add_edge("chips", "ai", EdgeType.EXPRESSES, 0.6)
    g.add_edge("ai", "services", EdgeType.SEEKS, 0.5)


def _openai(g: TopologicalGraph) -> None:
    g.add_node(label="GPT Models", node_id="gpt")
    g.add_node(label="Microsoft Partnership", node_id="microsoft")
    g.add_node(label="API Revenue", node_id="api_revenue")
    g.add_node(label="ChatGPT Consumer", node_id="chatgpt")
    g.add_node(label="Safety Team", node_id="safety")
    g.add_node(label="Compute Costs", node_id="compute")
    g.add_node(label="AGI Mission", node_id="agi_mission")
    g.add_node(label="Talent Retention", node_id="talent")
    g.add_node(label="Open Source Competition", node_id="open_source")
    g.add_node(label="Regulatory Scrutiny", node_id="regulation")
    g.add_node(label="Capped Profit Structure", node_id="capped_profit")
    g.add_node(label="Data Partnerships", node_id="data")

    g.add_edge("gpt", "chatgpt", EdgeType.EXPRESSES, 0.9)
    g.add_edge("gpt", "api_revenue", EdgeType.EXPRESSES, 0.8)
    g.add_edge("gpt", "compute", EdgeType.BOUNDS, 0.8)
    g.add_edge("microsoft", "compute", EdgeType.EXPRESSES, 0.9)
    g.add_edge("microsoft", "api_revenue", EdgeType.VERIFIES, 0.6)
    g.add_edge("chatgpt", "api_revenue", EdgeType.MIRRORS, 0.5)
    g.add_edge("safety", "agi_mission", EdgeType.REMOVES, 0.6)
    g.add_edge("safety", "regulation", EdgeType.MIRRORS, 0.5)
    g.add_edge("agi_mission", "talent", EdgeType.EXPRESSES, 0.7)
    g.add_edge("agi_mission", "capped_profit", EdgeType.REMOVES, 0.6)
    g.add_edge("compute", "api_revenue", EdgeType.REMOVES, 0.5)
    g.add_edge("talent", "open_source", EdgeType.SEEKS, 0.5)
    g.add_edge("open_source", "api_revenue", EdgeType.REMOVES, 0.6)
    g.add_edge("open_source", "gpt", EdgeType.REMOVES, 0.4)
    g.add_edge("regulation", "agi_mission", EdgeType.BOUNDS, 0.5)
    g.add_edge("capped_profit", "microsoft", EdgeType.SEEKS, 0.6)
    g.add_edge("data", "gpt", EdgeType.BOUNDS, 0.7)
    g.add_edge("data", "regulation", EdgeType.SEEKS, 0.5)
