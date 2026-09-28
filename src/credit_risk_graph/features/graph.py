from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(slots=True)
class GraphFeatureBundle:
    borrower_features: pd.DataFrame
    borrower_month_features: pd.DataFrame
    sampled_layout: pd.DataFrame


def build_training_graph_features(
    transactions: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    borrower_default_rates: pd.DataFrame,
) -> GraphFeatureBundle:
    tx_train = transactions.loc[
        transactions["tx_date"] <= train_cutoff,
        ["borrower_id", "counterparty_id", "counterparty_type", "tx_date", "amount"],
    ].copy()
    pair_edges = (
        tx_train.loc[tx_train["counterparty_type"] == "borrower", ["borrower_id", "counterparty_id", "amount"]]
        .groupby(["borrower_id", "counterparty_id"], as_index=False)["amount"]
        .sum()
    )
    graph = nx.Graph()
    graph.add_weighted_edges_from(pair_edges.itertuples(index=False, name=None))
    if graph.number_of_nodes() == 0:
        empty = pd.DataFrame(
            columns=[
                "borrower_id",
                "graph_degree",
                "graph_weighted_degree",
                "graph_pagerank",
                "graph_clustering",
                "graph_component_size",
                "graph_neighbor_default_rate",
                "graph_contagion_score",
                "graph_community_risk",
            ]
        )
        empty_month = pd.DataFrame(
            columns=[
                "borrower_id",
                "month_key",
                "graph_degree_3m",
                "graph_degree_6m",
                "graph_weighted_degree_3m",
                "graph_weighted_degree_6m",
                "graph_neighbor_default_rate_6m",
                "graph_neighbor_default_rate_12m",
                "graph_contagion_score_6m",
                "graph_counterparty_hhi_6m",
                "graph_temporal_weighted_degree",
                "graph_neighbor_community_risk_6m",
            ]
        )
        return GraphFeatureBundle(empty, empty_month, pd.DataFrame(columns=["node_id", "x", "y", "weighted_degree"]))

    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    pagerank = nx.pagerank(graph, weight="weight", alpha=0.9)
    clustering = nx.clustering(graph, weight="weight")
    component_lookup: dict[str, int] = {}
    for component in nx.connected_components(graph):
        comp_size = len(component)
        for node in component:
            component_lookup[node] = comp_size

    community_lookup: dict[str, int] = {}
    community_risk_lookup: dict[str, float] = {}
    communities = list(nx.community.asyn_lpa_communities(graph, weight="weight", seed=42))
    default_map = borrower_default_rates.set_index("borrower_id")["default_rate_train"].to_dict()
    for community_id, community in enumerate(communities):
        members = list(community)
        community_risk = float(np.mean([default_map.get(member, 0.0) for member in members])) if members else 0.0
        for node in members:
            community_lookup[node] = community_id
            community_risk_lookup[node] = community_risk

    neighbor_default_rate: dict[str, float] = {}
    contagion_score: dict[str, float] = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            neighbor_default_rate[node] = 0.0
            contagion_score[node] = 0.0
            continue
        neighbor_scores = np.array([default_map.get(neighbor, 0.0) for neighbor in neighbors], dtype=float)
        weights = np.array([graph.get_edge_data(node, neighbor).get("weight", 1.0) for neighbor in neighbors], dtype=float)
        weights = np.where(weights <= 0, 1.0, weights)
        neighbor_default_rate[node] = float(neighbor_scores.mean())
        contagion_score[node] = float(np.average(neighbor_scores, weights=weights))

    features = pd.DataFrame(
        {
            "borrower_id": list(graph.nodes()),
            "graph_degree": [degree[node] for node in graph.nodes()],
            "graph_weighted_degree": [weighted_degree[node] for node in graph.nodes()],
            "graph_pagerank": [pagerank[node] for node in graph.nodes()],
            "graph_clustering": [clustering[node] for node in graph.nodes()],
            "graph_component_size": [component_lookup[node] for node in graph.nodes()],
            "graph_neighbor_default_rate": [neighbor_default_rate[node] for node in graph.nodes()],
            "graph_contagion_score": [contagion_score[node] for node in graph.nodes()],
            "graph_community_risk": [community_risk_lookup.get(node, 0.0) for node in graph.nodes()],
        }
    )
    features["graph_degree"] = features["graph_degree"].astype(float)
    features["graph_component_size"] = features["graph_component_size"].astype(float)

    tx_monthly = tx_train.loc[tx_train["counterparty_type"] == "borrower", ["borrower_id", "counterparty_id", "tx_date", "amount"]].copy()
    tx_monthly["month_key"] = tx_monthly["tx_date"].dt.to_period("M").dt.to_timestamp()
    tx_monthly["counterparty_default_rate"] = tx_monthly["counterparty_id"].map(default_map).fillna(0.0)
    tx_monthly["counterparty_community_risk"] = tx_monthly["counterparty_id"].map(community_risk_lookup).fillna(0.0)
    pair_monthly = (
        tx_monthly.groupby(["borrower_id", "month_key", "counterparty_id"], as_index=False)
        .agg(
            monthly_amount=("amount", "sum"),
            counterparty_default_rate=("counterparty_default_rate", "max"),
            counterparty_community_risk=("counterparty_community_risk", "max"),
        )
    )
    concentration = (
        pair_monthly.groupby(["borrower_id", "month_key"], as_index=False)["monthly_amount"]
        .apply(lambda s: float(((s / max(s.sum(), 1.0)) ** 2).sum()))
        .rename(columns={"monthly_amount": "graph_counterparty_hhi_month"})
    )
    weighted_neighbor_risk = (
        pair_monthly.assign(
            weighted_default=lambda df: df["monthly_amount"] * df["counterparty_default_rate"],
            weighted_community_risk=lambda df: df["monthly_amount"] * df["counterparty_community_risk"],
        )
        .groupby(["borrower_id", "month_key"], as_index=False)
        .agg(
            graph_degree_month=("counterparty_id", "nunique"),
            graph_weighted_degree_month=("monthly_amount", "sum"),
            weighted_default_sum=("weighted_default", "sum"),
            weighted_community_risk_sum=("weighted_community_risk", "sum"),
        )
        .merge(concentration, on=["borrower_id", "month_key"], how="left")
    )
    weighted_neighbor_risk["graph_neighbor_default_rate_month"] = (
        weighted_neighbor_risk["weighted_default_sum"] / weighted_neighbor_risk["graph_weighted_degree_month"].clip(lower=1.0)
    )
    weighted_neighbor_risk["graph_neighbor_community_risk_month"] = (
        weighted_neighbor_risk["weighted_community_risk_sum"] / weighted_neighbor_risk["graph_weighted_degree_month"].clip(lower=1.0)
    )
    weighted_neighbor_risk["graph_contagion_score_month"] = (
        0.65 * weighted_neighbor_risk["graph_neighbor_default_rate_month"]
        + 0.35 * weighted_neighbor_risk["graph_neighbor_community_risk_month"]
    )
    weighted_neighbor_risk = weighted_neighbor_risk.sort_values(["borrower_id", "month_key"]).reset_index(drop=True)
    for source_col, target_col, window in [
        ("graph_degree_month", "graph_degree_3m", 3),
        ("graph_degree_month", "graph_degree_6m", 6),
        ("graph_weighted_degree_month", "graph_weighted_degree_3m", 3),
        ("graph_weighted_degree_month", "graph_weighted_degree_6m", 6),
        ("graph_neighbor_default_rate_month", "graph_neighbor_default_rate_6m", 6),
        ("graph_neighbor_default_rate_month", "graph_neighbor_default_rate_12m", 12),
        ("graph_contagion_score_month", "graph_contagion_score_6m", 6),
        ("graph_counterparty_hhi_month", "graph_counterparty_hhi_6m", 6),
        ("graph_neighbor_community_risk_month", "graph_neighbor_community_risk_6m", 6),
    ]:
        weighted_neighbor_risk[target_col] = (
            weighted_neighbor_risk.groupby("borrower_id")[source_col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        )
    weighted_neighbor_risk["graph_temporal_weighted_degree"] = (
        weighted_neighbor_risk["graph_weighted_degree_3m"] / (weighted_neighbor_risk["graph_weighted_degree_6m"] + 1.0)
    )
    borrower_month_features = weighted_neighbor_risk[
        [
            "borrower_id",
            "month_key",
            "graph_degree_3m",
            "graph_degree_6m",
            "graph_weighted_degree_3m",
            "graph_weighted_degree_6m",
            "graph_neighbor_default_rate_6m",
            "graph_neighbor_default_rate_12m",
            "graph_contagion_score_6m",
            "graph_counterparty_hhi_6m",
            "graph_temporal_weighted_degree",
            "graph_neighbor_community_risk_6m",
        ]
    ].fillna(0.0)

    layout_nodes = features.nlargest(250, "graph_weighted_degree")["borrower_id"].tolist()
    sampled_graph = graph.subgraph(layout_nodes).copy()
    layout = nx.spring_layout(sampled_graph, seed=42, weight="weight")
    sampled_layout = pd.DataFrame(
        {
            "node_id": list(layout.keys()),
            "x": [coord[0] for coord in layout.values()],
            "y": [coord[1] for coord in layout.values()],
            "weighted_degree": [weighted_degree[node] for node in layout.keys()],
        }
    )
    return GraphFeatureBundle(features, borrower_month_features, sampled_layout)
