# ATCM Topic Modeling and Network Analysis

This repository contains a series of Python scripts and resources designed to analyze papers submitted to the Antarctic Treaty Consultative Meetings (ATCMs). The code applies **topic modeling** (via Latent Dirichlet Allocation), **network analysis** (e.g., co-authorship networks, topic-specific collaborations, modularity detection), and **statistical validation** (via permutation tests) to uncover patterns of collaboration and topic prominence among contributing countries and actors.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
  - [AttHeatMaps.py](#attheatmapspy)
  - [Modularity.py](#modularitypy)
  - [Permutations.py](#permutationspy)
  - [TopicModelReplica.py](#topicmodelreplicapy)
  - [Visualizations.py](#visualizationspy)
  - [Antarctic_Governance_Methods_Parsa.pdf](#antarctic_governance_methods_parsapdf)

---

## Project Overview

Since 1991, the Antarctic Treaty Consultative Meetings have seen a steady stream of research papers and policy documents focusing on Antarctic governance, environmental protection, science, and international collaboration. This project seeks to:

1. **Identify major topical themes** across the ATCM submissions using Latent Dirichlet Allocation (LDA).
2. **Map out collaboration networks** (who co-authors with whom, how topics connect different countries/organizations).
3. **Statistically test** whether certain actor attributes are significantly associated with higher network centrality, more frequent submissions on specific topics, etc.

Ultimately, this analysis aims to provide insight into **Antarctic governance processes**, showing how different stakeholders collaborate and which topics garner the most attention.

---

## Repository Structure

Below is an overview of each file in the repository:

### AttHeatMaps.py
Generates **correlation heatmaps** between actor attributes and network metrics. It:
- Merges an attributes DataFrame with a topic/network-metrics DataFrame.
- Creates a correlation matrix for each attribute/metric pair.
- Produces annotated heatmaps illustrating positive/negative correlations.

### Modularity.py
Conducts **community detection** (primarily using the Louvain algorithm) and **hierarchical subdivision** of a network. It:
- Loads or constructs a network from CSV inputs.
- Applies Louvain-based modularity optimization to detect communities.
- Calculates advanced metrics like AIC and silhouette scores to evaluate community quality.
- Includes a permutation-based statistical significance test for modularity values.

### Permutations.py
Implements **permutation tests** to assess significance of relationships between participant attributes (binary or categorical) and network metrics. It:
- Merges attribute data with network metrics.
- Runs two-sample permutation tests for binary variables and an ANOVA-like permutation approach for multi-category variables.
- Adjusts p-values for multiple comparisons.

### TopicModelReplica.py
Performs **topic modeling** using LDA. It:
- Cleans and preprocesses the text (removing punctuation, lemmatizing, etc.).
- Builds a document-term matrix.
- Fits an LDA model (with adjustable number of topics).
- Outputs both the topic-word distributions and the per-document topic scores.

### Visualizations.py
Provides **network and data visualization** utilities, such as:
- Building bipartite networks of topics vs. submitters.
- Visualizing co-authorship networks with PyVis or NetworkX.
- Generating heatmaps, bar charts, or other plots to show top collaborators, highest topic scorers, etc.


