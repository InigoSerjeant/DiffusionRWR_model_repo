from pathlib import Path
import io
import json
import zipfile
import re
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from DiffusionRWR_model_package.diffusion_functions.clustering_analysis import (
    cluster_multiple_graphs,
    calculate_pairwise_mutual_information,
)
from DiffusionRWR_model_package.model_analysis.app_math import (
    is_integer_like as pkg_is_integer_like,
    standardize_rows as pkg_standardize_rows,
    build_adjacency as pkg_build_adjacency,
    graph_from_adjacency as pkg_graph_from_adjacency,
    graph_from_adjacency_with_forced_nodes as pkg_graph_from_adjacency_with_forced_nodes,
    compute_eigenvector_centrality as pkg_compute_eigenvector_centrality,
    compute_pagerank_centrality as pkg_compute_pagerank_centrality,
    compute_degree_centrality as pkg_compute_degree_centrality,
    closest_nodes_to_basis as pkg_closest_nodes_to_basis,
    build_multigraph_adjacency_for_clustering as pkg_build_multigraph_adjacency_for_clustering,
    simulate_diffusion_visits,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DiffusionRWR_model_package" / "data" / "Modelled"
OUTPUT_DIR = ROOT / "DiffusionRWR_model_package" / "outputs"


def _discover_lasso_combined_source_map() -> dict[str, tuple[str, str]]:
    pattern = re.compile(r"^overlap_filtered_(.+)_m_v2\.csv$", re.IGNORECASE)
    source_map: dict[str, tuple[str, str]] = {}

    if not DATA_DIR.exists():
        return source_map

    for path in sorted(DATA_DIR.glob("overlap_filtered_*_m_v2.csv")):
        match = pattern.match(path.name)
        if not match:
            continue

        token = match.group(1)
        if token.lower().startswith("rna"):
            suffix = "RNA"
            label = "RNA"
        else:
            suffix = token[:1].upper() + token[1:]
            label = suffix

        if label in source_map:
            alt_label = f"{label} ({path.stem})"
            source_map[alt_label] = (path.name, suffix)
        else:
            source_map[label] = (path.name, suffix)

    return source_map


LASSO_COMBINED_SOURCE_MAP = _discover_lasso_combined_source_map()


def _discover_histone_dataset_options(folder_path: str) -> list[str]:
    modelled_dir = Path(folder_path)
    if not modelled_dir.exists():
        return []

    token_pattern = re.compile(r"^overlap_filtered_(.+)_m_v2\.csv$", re.IGNORECASE)
    options: list[str] = []
    for path in sorted(modelled_dir.glob("overlap_filtered_*_m_v2.csv")):
        match = token_pattern.match(path.name)
        if not match:
            continue
        token = match.group(1).lower()
        if token.startswith("rna"):
            continue
        if token not in options:
            options.append(token)
    return options


def _discover_histone_start_nodes(folder_path: str, histone_token: str) -> list[str]:
    modelled_dir = Path(folder_path)
    if not modelled_dir.exists() or not histone_token:
        return []

    target_name = f"overlap_filtered_{histone_token}_m_v2.csv"
    csv_path = modelled_dir / target_name
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if len(df.columns) > 0 and (df.columns[0] == "Unnamed: 0" or "gene" in df.columns[0].lower()):
        genes = df[df.columns[0]].dropna().astype(str).tolist()
    else:
        genes = df.index.astype(str).tolist()

    return sorted([gene for gene in genes if gene and not gene.startswith("e")])


def _json_safe(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Index):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _build_export_zip(
    base_name: str,
    params: dict,
    plotly_fig: go.Figure | None = None,
    html_content: str | None = None,
    csv_frames: dict[str, pd.DataFrame] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if plotly_fig is not None:
            zf.writestr(f"{base_name}.html", plotly_fig.to_html(full_html=True, include_plotlyjs="cdn"))
        if html_content is not None:
            zf.writestr(f"{base_name}.html", html_content)
        if csv_frames:
            for frame_name, frame_df in csv_frames.items():
                zf.writestr(f"{frame_name}.csv", frame_df.to_csv(index=False))
        zf.writestr("parameters.json", json.dumps(_json_safe(params), indent=2, ensure_ascii=False))
    buffer.seek(0)
    return buffer.getvalue()


def _slugify_label(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value.strip())
    slug = slug.strip("_").lower()
    return slug or "plot"


def _plotly_chart_with_download(
    fig: go.Figure,
    *,
    base_name: str,
    button_label: str,
    key: str,
    params: dict,
    csv_frames: dict[str, pd.DataFrame] | None = None,
):
    st.plotly_chart(fig, use_container_width=True)
    safe_base = _slugify_label(base_name)
    export_zip = _build_export_zip(
        base_name=safe_base,
        params=params,
        plotly_fig=fig,
        csv_frames=csv_frames,
    )
    st.download_button(
        button_label,
        data=export_zip,
        file_name=f"{safe_base}_export.zip",
        mime="application/zip",
        key=key,
    )


def _show_graph_hyperparameters(title: str, params: dict):
    with st.expander(title, expanded=False):
        st.json(_json_safe(params), expanded=False)


def _show_rendered_node_selection_explanation(include_forced_nodes: bool = False):
    explanation = (
        "Rendered-node subsetting: after edge-thresholding, node strength is computed as "
        "row-sum + column-sum of the thresholded adjacency; if this exceeds the max-node cap, "
        "the top-strength nodes are kept and the graph is induced on that subset."
    )
    if include_forced_nodes:
        explanation += " In this view, forced nodes (for example temporal basis nodes) are always included."
    explanation += " Isolated nodes are then removed in standard graph rendering."
    st.caption(explanation)


@st.cache_data
def discover_csv_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.csv"))


@st.cache_data
def load_dataset(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_csv(path)
    if len(df.columns) > 0 and (df.columns[0] == "Unnamed: 0" or "gene" in df.columns[0].lower()):
        df = df.set_index(df.columns[0])

    integer_cols = [c for c in df.columns if _is_integer_like(c)]
    if integer_cols:
        df = df[integer_cols]

    return df


@st.cache_data
def standardize_rows(df: pd.DataFrame) -> pd.DataFrame:
    return pkg_standardize_rows(df)


def _is_integer_like(value: str) -> bool:
    return pkg_is_integer_like(value)


def build_adjacency(df: pd.DataFrame, method: str, n_power: int, sigma: float) -> pd.DataFrame:
    return pkg_build_adjacency(df, method, n_power, sigma)


def graph_from_adjacency(adjacency: pd.DataFrame, threshold: float, max_nodes: int) -> nx.DiGraph:
    return pkg_graph_from_adjacency(adjacency, threshold, max_nodes)


def graph_from_adjacency_with_forced_nodes(
    adjacency: pd.DataFrame,
    threshold: float,
    max_nodes: int,
    forced_nodes: list[str],
) -> nx.DiGraph:
    return pkg_graph_from_adjacency_with_forced_nodes(adjacency, threshold, max_nodes, forced_nodes)


def _load_modelled_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) > 0 and (df.columns[0] == "Unnamed: 0" or "gene" in df.columns[0].lower()):
        df = df.set_index(df.columns[0])
    integer_cols = [c for c in df.columns if _is_integer_like(c)]
    if integer_cols:
        df = df[integer_cols]
    return df


@st.cache_data(show_spinner=False)
def load_combined_standardized() -> pd.DataFrame:
    return load_combined_standardized_selected(tuple(LASSO_COMBINED_SOURCE_MAP.keys()))


@st.cache_data(show_spinner=False)
def load_combined_standardized_selected(selected_sources: tuple[str, ...]) -> pd.DataFrame:
    if not selected_sources:
        return pd.DataFrame()

    frames = []
    for source_name in selected_sources:
        if source_name not in LASSO_COMBINED_SOURCE_MAP:
            continue
        filename, suffix = LASSO_COMBINED_SOURCE_MAP[source_name]
        source_path = DATA_DIR / filename
        if not source_path.exists():
            continue
        source_df = _load_modelled_csv(source_path)
        source_df.index = [f"{gene}_{suffix}" for gene in source_df.index]
        frames.append(source_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0)
    combined = combined.sort_index()
    combined_std = combined.sub(combined.mean(axis=1), axis=0).div(combined.std(axis=1), axis=0)
    return combined_std


def build_lasso_adjacency_from_standardized(df_standardized: pd.DataFrame, key: str, lasso_alpha: float = 0.01) -> pd.DataFrame:
    from DiffusionRWR_model_package.graph_generation.generate_graph_internal import lasso_single_graph

    if df_standardized.empty:
        return pd.DataFrame()

    gene_names = [name for name in df_standardized.index if not (name.startswith('e') and len(name) == 2)]
    if len(gene_names) >= 2:
        start_gene = gene_names[0]
        end_gene = gene_names[-1]
    else:
        start_gene = df_standardized.index[0]
        end_gene = df_standardized.index[-1]

    adjacency_dict = lasso_single_graph(
        {key: df_standardized},
        edge_fn=None,
        start=start_gene,
        end=end_gene,
        return_signs=False,
        lasso_alpha=lasso_alpha,
    )
    return adjacency_dict[key].abs()


@st.cache_data(show_spinner=True)
def load_combined_lasso_adjacency(lasso_alpha: float = 0.01) -> pd.DataFrame:
    from DiffusionRWR_model_package.graph_generation.generate_graph_internal import lasso_single_graph

    combined_std = load_combined_standardized()
    if combined_std.empty:
        return pd.DataFrame()

    # Get gene names (exclude basis vectors like e1, e2, e3, e4, e5)
    gene_names = [name for name in combined_std.index if not (name.startswith('e') and len(name) == 2)]
    
    if len(gene_names) >= 2:
        start_gene = gene_names[0]
        end_gene = gene_names[-1]
    else:
        # Fallback: use all available node names in order
        start_gene = combined_std.index[0]
        end_gene = combined_std.index[-1]

    adjacency_dict = lasso_single_graph(
        {"combined_rna_histones": combined_std},
        edge_fn=None,
        start=start_gene,
        end=end_gene,
        return_signs=False,
        lasso_alpha=lasso_alpha,
    )
    return adjacency_dict["combined_rna_histones"]


def build_multigraph_adjacency_for_clustering(
    std_data_dict: dict[str, pd.DataFrame],
    intra_layer_graphs: dict[str, pd.DataFrame],
    alpha: float,
    inter_method: str,
    inter_n_power: int,
    inter_sigma: float,
    inter_layer_threshold: float,
) -> pd.DataFrame:
    return pkg_build_multigraph_adjacency_for_clustering(
        std_data_dict=std_data_dict,
        intra_layer_graphs=intra_layer_graphs,
        alpha=alpha,
        inter_method=inter_method,
        inter_n_power=inter_n_power,
        inter_sigma=inter_sigma,
        inter_layer_threshold=inter_layer_threshold,
    )


def _node_group(node_name: str) -> str:
    if node_name.startswith("e") and len(node_name) == 2:
        return "basis"
    upper = node_name.upper()
    if upper.endswith("_RNA"):
        return "rna"
    suffix = upper.rsplit("_", 1)[-1] if "_" in upper else ""
    if suffix.startswith("K") and "ME" in suffix:
        return "histone"
    return "other"


def compute_eigenvector_centrality(graph: nx.DiGraph) -> dict[str, float]:
    return pkg_compute_eigenvector_centrality(graph)


def compute_pagerank_centrality(graph: nx.DiGraph, alpha: float = 0.85) -> dict[str, float]:
    return pkg_compute_pagerank_centrality(graph, alpha=alpha)


def compute_degree_centrality(graph: nx.DiGraph) -> dict[str, float]:
    return pkg_compute_degree_centrality(graph)


def _closest_nodes_to_basis(graph: nx.DiGraph, pos: dict, basis_nodes: list[str]) -> dict[str, str]:
    return pkg_closest_nodes_to_basis(graph, pos, basis_nodes)


def plot_combined_temporal_graph(
    graph: nx.DiGraph,
    basis_nodes: list[str],
    centrality_type: str = "degree",
    pagerank_alpha: float = 0.85,
) -> tuple[go.Figure, dict[str, str]]:
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No nodes after thresholding")
        return fig, {}

    # Compute centrality based on user choice
    if centrality_type == "eigenvector":
        centrality = compute_eigenvector_centrality(graph)
    elif centrality_type == "pagerank":
        centrality = compute_pagerank_centrality(graph, alpha=pagerank_alpha)
    else:  # degree
        centrality = compute_degree_centrality(graph)

    pos = nx.spring_layout(graph, seed=42, k=0.5)

    basis_color_map = {
        "e1": "gold",
        "e2": "orange",
        "e3": "red",
        "e4": "purple",
        "e5": "darkblue",
    }

    closest_to_basis = _closest_nodes_to_basis(graph, pos, basis_nodes)
    anchor_nodes = set(closest_to_basis.values())

    edge_x = []
    edge_y = []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="rgba(120,120,120,0.35)"),
        hoverinfo="none",
        showlegend=False,
    )

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []

    default_type_colors = {
        "rna": "lightblue",
        "histone": "lightcoral",
        "other": "gray",
        "basis": "black",
    }

    for node in graph.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Calculate size based on centrality
        cent_value = centrality.get(node, 0.0)
        base_size = 5 + 15 * np.sqrt(max(cent_value, 0.0) / max(centrality.values()))

        if node in basis_nodes:
            color = basis_color_map.get(node, "black")
            size = max(base_size, 18)  # Ensure basis nodes are at least size 18
        elif node in anchor_nodes:
            owner_basis = next((b for b, n in closest_to_basis.items() if n == node), None)
            color = basis_color_map.get(owner_basis, "black")
            size = max(base_size, 12)  # Ensure anchor nodes are at least size 12
        else:
            color = default_type_colors.get(_node_group(node), "gray")
            size = base_size

        node_color.append(color)
        node_size.append(size)
        node_text.append(
            f"{node}<br>out={graph.out_degree(node)}, in={graph.in_degree(node)}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=node_size, color=node_color, opacity=0.92),
        text=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=(
            "Combined Lasso graph with temporal anchors "
            f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=760,
    )
    return fig, closest_to_basis


def plot_graph_2d(graph: nx.DiGraph, centrality_type: str = "degree", pagerank_alpha: float = 0.85) -> go.Figure:
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No nodes after thresholding")
        return fig

    # Compute centrality based on user choice
    if centrality_type == "eigenvector":
        centrality = compute_eigenvector_centrality(graph)
    elif centrality_type == "pagerank":
        centrality = compute_pagerank_centrality(graph, alpha=pagerank_alpha)
    else:  # degree
        centrality = compute_degree_centrality(graph)

    pos = nx.spring_layout(graph, seed=42, k=0.5)

    edge_x = []
    edge_y = []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="rgba(140,140,140,0.45)"),
        hoverinfo="none",
        showlegend=False,
    )

    node_x = []
    node_y = []
    node_size = []
    node_text = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        out_d = graph.out_degree(node)
        in_d = graph.in_degree(node)
        
        # Size based on centrality
        cent_value = centrality.get(node, 0.0)
        if max(centrality.values()) > 0:
            normalized_cent = cent_value / max(centrality.values())
        else:
            normalized_cent = 0.0
        size = 6 + 20 * np.sqrt(normalized_cent)
        
        node_size.append(size)
        node_text.append(f"{node}<br>out={out_d}, in={in_d}<br>centrality={cent_value:.4f}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=node_size, color=node_size, colorscale="Viridis", opacity=0.9),
        text=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Network graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=700,
    )
    return fig


def show_overview():
    st.title("DiffusionRWR Maths + Plot Visualiser")
    st.write(
        "Interactive dashboard for exploring datasets, equations, generated outputs, and custom graph projections from DiffusionRWR_model_repo."
    )

    csv_files = discover_csv_files()
    c1, c2, c3 = st.columns(3)
    c1.metric("Modelled CSV files", len(csv_files))
    c2.metric("Output directory exists", "Yes" if OUTPUT_DIR.exists() else "No")
    c3.metric("Repo root", ROOT.name)

    if csv_files:
        st.subheader("Available datasets")
        st.dataframe(pd.DataFrame({"file": [p.name for p in csv_files]}), use_container_width=True)



def show_math_page():
    st.title("Maths behind the package")

    st.subheader("Core correlation")
    st.latex(r"r_{ij} = \mathrm{corr}(x_i, x_j)")

    st.subheader("Power edge weighting")
    st.latex(r"s_{ij} = \tfrac{1}{2}(r_{ij}+1)")
    st.latex(r"w_{ij} = |s_{ij}|^{n_{\mathrm{power}}}")

    st.subheader("Absolute-correlation Gaussian weighting")
    st.latex(r"w_{ij} = \exp\left(-\tfrac{1}{2}\left(\frac{1-|r_{ij}|}{\sigma}\right)^2\right)")

    st.subheader("Absolute-correlation Exponential weighting")
    st.latex(r"w_{ij} = \exp\left(-\frac{1-|r_{ij}|}{\sigma}\right)")

    st.subheader("Row-wise z-score used in preprocessing")
    st.latex(r"z_{i,t} = \frac{x_{i,t} - \mu_i}{\sigma_i}")

    st.subheader("Lasso graph construction concept")
    st.latex(r"\hat{\beta} = \arg\min_{\beta} \left\{\frac{1}{2n}\|y - X\beta\|_2^2 + \alpha\|\beta\|_1\right\}")
    st.write("Each target node trajectory is regressed against all other node trajectories; non-zero coefficients define directed edges.")

    st.subheader("Random Walk with Restart (RWR)")
    st.write("**Transition Matrix Construction:**")
    st.latex(r"P_{ij} = \frac{w_{ij}}{\sum_k w_{ik}}")
    st.write("For Lasso graphs (unweighted), all edges equally likely: $ P_{ij} = \\frac{1}{d_i^{out}}$ where $d_i^{out}$ is out-degree")
    st.write("For correlation graphs, edge weights determine transition probabilities")

    st.write("**Single Random Walk Step:**")
    st.latex(r"X_t \rightarrow X_{t+1} \begin{cases} X_s & \text{with probability } r \\ X_j & \text{with probability } (1-r) \text{ where } j \sim P_{X_t} \end{cases}")
    st.write("where $r$ is the restart probability, $s$ is the start node, and $P_{X_t}$ is the transition distribution from current node")

    st.write("**Multiple Walks & Visit Frequency:**")
    st.latex(r"v_i = \frac{1}{n_{\mathrm{walks}} \times n_{\mathrm{steps}}} \sum_{k=1}^{n_{\mathrm{walks}}} \sum_{t=1}^{n_{\mathrm{steps}}} \mathbb{1}(X_t^{(k)} = i)")
    st.write("where $\\mathbb{1}(X_t^{(k)} = i)$ indicates whether walk $k$ visits node $i$ at step $t$")

    st.write("**Normalized Visit Frequency (Centrality):**")
    st.latex(r"c_i = \frac{v_i}{\sum_j v_j}")


def show_data_page():
    st.title("Dataset explorer")
    csv_files = discover_csv_files()
    if not csv_files:
        st.error(f"No CSV files found in: {DATA_DIR}")
        return

    selected = st.selectbox("Choose dataset", options=csv_files, format_func=lambda p: p.name)
    dataset_name = selected.name
    dataset_slug = _slugify_label(selected.stem)
    df = load_dataset(str(selected))

    st.write(f"Shape: {df.shape[0]} genes × {df.shape[1]} time points")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Missingness and summary")
    c1, c2 = st.columns(2)
    c1.metric("Missing cells", int(df.isna().sum().sum()))
    c2.metric("Numeric columns", df.select_dtypes(include=[np.number]).shape[1])

    st.subheader("Distribution snapshot")
    flattened = df.select_dtypes(include=[np.number]).to_numpy().ravel()
    flattened = flattened[np.isfinite(flattened)]
    if flattened.size > 0:
        fig = px.histogram(flattened, nbins=60, title=f"Value distribution: {selected.name}")
        hist_params = {
            "page": "Dataset Explorer",
            "dataset": dataset_name,
            "nbins": 60,
        }
        _plotly_chart_with_download(
            fig,
            base_name=f"{dataset_slug}_value_distribution",
            button_label="Download value distribution + parameters",
            key="download_dataset_histogram",
            params=hist_params,
        )

    st.subheader("Correlation heatmap (top genes by variance)")
    top_n = st.slider("Top genes by variance", 20, 200, 60, 10)
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.empty:
        st.info("No numeric data available for heatmap.")
        return
    variances = num_df.var(axis=1).sort_values(ascending=False)
    picked = variances.head(min(top_n, len(variances))).index
    corr = num_df.loc[picked].T.corr()
    heat = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1)
    heat.update_layout(height=700)
    heat_params = {
        "page": "Dataset Explorer",
        "dataset": dataset_name,
        "top_genes": int(top_n),
    }
    _plotly_chart_with_download(
        heat,
        base_name=f"{dataset_slug}_correlation_heatmap",
        button_label="Download correlation heatmap + parameters",
        key="download_dataset_heatmap",
        params=heat_params,
    )



def show_graph_builder_page():
    st.title("Interactive graph builder")
    csv_files = discover_csv_files()
    if not csv_files:
        st.error(f"No CSV files found in: {DATA_DIR}")
        return

    selected = st.selectbox("Dataset", options=csv_files, format_func=lambda p: p.name)
    dataset_slug = _slugify_label(selected.stem)
    raw_df = load_dataset(str(selected))
    df = standardize_rows(raw_df)

    st.subheader("Edge weighting")
    method = st.selectbox("Method", ["corr_power", "cor_gaussian_abs", "cor_exponential_abs"])
    n_power = st.slider("Power n (corr_power only)", 2, 40, 20)
    sigma = st.slider("Sigma", 0.001, 1.0, 0.05, 0.001)

    st.subheader("Thresholding + node cap")
    threshold = st.slider("Edge threshold", 0.0, 1.0, 0.15, 0.01)
    max_nodes = st.slider("Max nodes to render", 50, 3000, 300, 50)
    _show_rendered_node_selection_explanation(include_forced_nodes=False)

    graph_builder_params = {
        "page": "Graph Builder",
        "dataset": selected.name,
        "method": method,
        "n_power": int(n_power),
        "sigma": float(sigma),
        "threshold": float(threshold),
        "max_nodes": int(max_nodes),
    }
    _show_graph_hyperparameters("Graph generation hyperparameters", graph_builder_params)

    generate_graph = st.button("Generate graph", key="graph_builder_generate_button")
    graph_signature = (str(selected), method, int(n_power), float(sigma), float(threshold), int(max_nodes))

    if generate_graph:
        adjacency = build_adjacency(df, method=method, n_power=n_power, sigma=sigma)
        graph = graph_from_adjacency(adjacency, threshold=threshold, max_nodes=max_nodes)
        st.session_state["graph_builder_graph_signature"] = graph_signature
        st.session_state["graph_builder_graph_data"] = nx.node_link_data(graph)
        st.success("Graph generated.")

    if st.session_state.get("graph_builder_graph_signature") != graph_signature:
        st.info("Adjust parameters and click 'Generate graph' to update the graph.")
        return

    if "graph_builder_graph_data" not in st.session_state:
        st.info("Click 'Generate graph' to build the graph.")
        return

    graph = nx.node_link_graph(st.session_state["graph_builder_graph_data"], directed=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rendered nodes", graph.number_of_nodes())
    c2.metric("Rendered edges", graph.number_of_edges())
    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    c3.metric("Density", f"{density:.4f}")

    st.subheader("Centrality visualization")
    centrality_choice = st.selectbox(
        "Node size based on",
        ["degree", "eigenvector", "pagerank"],
        key="graph_builder_centrality"
    )
    
    pagerank_alpha = 0.85
    if centrality_choice == "pagerank":
        pagerank_alpha = st.slider(
            "PageRank alpha (damping factor)",
            0.1, 0.99, 0.85, 0.05,
            key="graph_builder_pagerank_alpha"
        )

    fig = plot_graph_2d(graph, centrality_type=centrality_choice, pagerank_alpha=pagerank_alpha)
    graph_builder_plot_params = {
        **graph_builder_params,
        "centrality": centrality_choice,
        "pagerank_alpha": float(pagerank_alpha) if centrality_choice == "pagerank" else None,
    }
    _plotly_chart_with_download(
        fig,
        base_name=f"{dataset_slug}_graph_builder",
        button_label="Download graph visualization + parameters",
        key="download_graph_builder_plot",
        params=graph_builder_plot_params,
    )

    with st.expander("Show top hubs"):
        if graph.number_of_nodes() == 0:
            st.info("No graph nodes available.")
        else:
            out_deg = sorted(graph.out_degree, key=lambda x: x[1], reverse=True)[:20]
            in_deg = sorted(graph.in_degree, key=lambda x: x[1], reverse=True)[:20]
            left, right = st.columns(2)
            left.dataframe(pd.DataFrame(out_deg, columns=["node", "out_degree"]), use_container_width=True)
            right.dataframe(pd.DataFrame(in_deg, columns=["node", "in_degree"]), use_container_width=True)
    
    with st.expander("Show top centrality"):
        if graph.number_of_nodes() == 0:
            st.info("No graph nodes available.")
        else:
            # Compute selected centrality
            if centrality_choice == "eigenvector":
                centrality = compute_eigenvector_centrality(graph)
            elif centrality_choice == "pagerank":
                centrality = compute_pagerank_centrality(graph, alpha=pagerank_alpha)
            else:
                centrality = compute_degree_centrality(graph)
            
            top_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
            st.dataframe(
                pd.DataFrame(top_centrality, columns=["node", f"{centrality_choice}_centrality"]),
                use_container_width=True
            )


def show_multigraph_builder_page():
    st.title("Interactive multigraph builder")
    csv_files = discover_csv_files()
    if not csv_files:
        st.error(f"No CSV files found in: {DATA_DIR}")
        return

    st.subheader("Multigraph configuration")
    c1, c2 = st.columns(2)
    with c1:
        graph_topology = st.selectbox(
            "Topology",
            ["Multi-layer", "Shadow multigraph"],
            key="multigraph_builder_topology",
        )
        graph_topology_slug = _slugify_label(graph_topology)
    with c2:
        selected_datasets = st.multiselect(
            "Datasets",
            options=[p.name for p in csv_files],
            default=[p.name for p in csv_files],
            key="multigraph_builder_datasets",
            help="Pick at least two datasets to form layers.",
        )

    if len(selected_datasets) < 2:
        st.warning("Select at least two datasets to build a multigraph.")
        return

    st.subheader("Intra-layer edge weighting")
    i1, i2, i3 = st.columns(3)
    with i1:
        method = st.selectbox(
            "Method",
            ["corr_power", "cor_gaussian_abs", "cor_exponential_abs"],
            key="multigraph_builder_intra_method",
        )
    with i2:
        n_power = st.slider("Power n (corr_power)", 2, 40, 20, key="multigraph_builder_intra_n_power")
    with i3:
        sigma = st.slider("Sigma (gaussian/exponential)", 0.001, 1.0, 0.05, 0.001, key="multigraph_builder_intra_sigma")

    st.subheader("Inter-layer weighting + transitions")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        inter_method = st.selectbox(
            "Inter-layer method",
            [
                "inter_layer_corr_power",
                "cor_exponential_abs_inter",
                "cor_gaussian_abs_inter",
                "intra_layer_corr_exponential_shifted",
                "intra_layer_corr_gaussian_shifted",
            ],
            key="multigraph_builder_inter_method",
        )
    with t2:
        inter_n_power = st.slider("Inter n_power", 2, 40, 20, key="multigraph_builder_inter_n_power")
    with t3:
        inter_sigma = st.slider("Inter sigma", 0.001, 1.0, 0.05, 0.001, key="multigraph_builder_inter_sigma")
    with t4:
        alpha = st.slider("Alpha (inter-layer transition)", 0.0, 1.0, 0.1, 0.01, key="multigraph_builder_alpha")
    with t5:
        inter_layer_threshold = st.slider(
            "Inter-layer threshold",
            0.0,
            1.0,
            0.0,
            0.001,
            key="multigraph_builder_inter_layer_threshold",
            help="Inter-layer edge weights below this are set to zero before graph assembly.",
        )

    gamma = st.slider(
        "Gamma (negative jump probability)",
        0.0,
        1.0,
        0.01,
        0.01,
        key="multigraph_builder_gamma",
        help="Used only for Shadow multigraph.",
    )
    if graph_topology != "Shadow multigraph":
        st.caption("Gamma is only applied for Shadow multigraph topology.")

    st.subheader("Thresholding + node cap")
    p1, p2 = st.columns(2)
    with p1:
        threshold = st.slider(
            "Edge threshold",
            0.0,
            0.05,
            0.0,
            0.001,
            key="multigraph_builder_threshold",
            help="Multigraph edges are probability-scaled; keep threshold low.",
        )
    with p2:
        max_nodes = st.slider("Max nodes to render", 50, 3000, 300, 50, key="multigraph_builder_max_nodes")
    _show_rendered_node_selection_explanation(include_forced_nodes=False)

    multigraph_generation_params = {
        "page": "Multigraph Builder",
        "topology": graph_topology,
        "selected_datasets": selected_datasets,
        "intra_method": method,
        "intra_n_power": int(n_power),
        "intra_sigma": float(sigma),
        "inter_method": inter_method,
        "inter_n_power": int(inter_n_power),
        "inter_sigma": float(inter_sigma),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "inter_layer_threshold": float(inter_layer_threshold),
        "edge_threshold": float(threshold),
        "max_nodes": int(max_nodes),
    }
    _show_graph_hyperparameters("Graph generation hyperparameters", multigraph_generation_params)

    generate_multigraph = st.button(
        "Generate all graph generations",
        key="multigraph_builder_generate_button",
    )

    graph = st.session_state.get("multigraph_builder_graph")

    if generate_multigraph:
        try:
            from DiffusionRWR_model_package.graph_generation import edge_weight_functions
            from DiffusionRWR_model_package.graph_generation.generate_multi_graph import (
                create_multigraph_with_layer_transitions,
                create_shadow_network_multigraph,
            )

            data_dict = {}
            for csv_file in csv_files:
                if csv_file.name in selected_datasets:
                    data_dict[csv_file.stem] = standardize_rows(load_dataset(str(csv_file)))

            if not data_dict:
                st.error("No datasets selected.")
                return

            intra_layer_graphs = {}
            sign_matrices = {}
            for dataset_name, std_df in data_dict.items():
                layer_adj = build_adjacency(std_df, method=method, n_power=n_power, sigma=sigma)
                intra_layer_graphs[dataset_name] = layer_adj

                corr_matrix = std_df.T.corr()
                sign_matrices[dataset_name] = pd.DataFrame(
                    np.sign(corr_matrix.values), index=corr_matrix.index, columns=corr_matrix.columns
                ).fillna(0)

            edge_fn_base = getattr(edge_weight_functions, inter_method)

            def edge_fn_inter(vec_a, vec_b):
                if inter_method == "inter_layer_corr_power":
                    weight = edge_fn_base(vec_a, vec_b, n_power=inter_n_power)
                elif inter_method in {
                    "cor_exponential_abs_inter",
                    "cor_gaussian_abs_inter",
                    "intra_layer_corr_exponential_shifted",
                    "intra_layer_corr_gaussian_shifted",
                }:
                    weight = edge_fn_base(vec_a, vec_b, sigma=inter_sigma)
                else:
                    weight = edge_fn_base(vec_a, vec_b)
                return weight if weight >= inter_layer_threshold else 0.0

            if graph_topology == "Shadow multigraph":
                multigraph_adjacency = create_shadow_network_multigraph(
                    std_data_dict=data_dict,
                    intra_graphs=intra_layer_graphs,
                    sign_matrices=sign_matrices,
                    edge_fn_inter=edge_fn_inter,
                    gamma=gamma,
                    alpha=alpha,
                    start="e1",
                    end="e5",
                )
            else:
                multigraph_adjacency = create_multigraph_with_layer_transitions(
                    std_data_dict=data_dict,
                    intra_layer_graphs=intra_layer_graphs,
                    edge_fn_inter=edge_fn_inter,
                    alpha=alpha,
                    start="e1",
                    end="e5",
                )

            graph = graph_from_adjacency(multigraph_adjacency, threshold=threshold, max_nodes=max_nodes)
            st.session_state["multigraph_builder_graph"] = graph
            st.success("Multigraph generated.")
        except Exception as e:
            st.error(f"Multigraph construction failed: {str(e)}")
            return

    if graph is None:
        st.info("Configure sliders and click 'Generate all graph generations' to build the multigraph.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rendered nodes", graph.number_of_nodes())
    c2.metric("Rendered edges", graph.number_of_edges())
    density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    c3.metric("Density", f"{density:.4f}")

    st.subheader("Centrality visualization")
    centrality_choice = st.selectbox(
        "Node size based on",
        ["degree", "eigenvector", "pagerank"],
        key="multigraph_builder_centrality",
    )

    pagerank_alpha = 0.85
    if centrality_choice == "pagerank":
        pagerank_alpha = st.slider(
            "PageRank alpha (damping factor)",
            0.1,
            0.99,
            0.85,
            0.05,
            key="multigraph_builder_pagerank_alpha",
        )

    fig = plot_graph_2d(graph, centrality_type=centrality_choice, pagerank_alpha=pagerank_alpha)
    multigraph_params = {
        **multigraph_generation_params,
        "centrality": centrality_choice,
        "pagerank_alpha": float(pagerank_alpha) if centrality_choice == "pagerank" else None,
    }
    _plotly_chart_with_download(
        fig,
        base_name=f"multigraph_{graph_topology_slug}",
        button_label="Download multigraph visualization + parameters",
        key="download_multigraph_builder_plot",
        params=multigraph_params,
    )

    with st.expander("Show top hubs"):
        if graph.number_of_nodes() == 0:
            st.info("No graph nodes available.")
        else:
            out_deg = sorted(graph.out_degree, key=lambda x: x[1], reverse=True)[:20]
            in_deg = sorted(graph.in_degree, key=lambda x: x[1], reverse=True)[:20]
            left, right = st.columns(2)
            left.dataframe(pd.DataFrame(out_deg, columns=["node", "out_degree"]), use_container_width=True)
            right.dataframe(pd.DataFrame(in_deg, columns=["node", "in_degree"]), use_container_width=True)

    with st.expander("Show top centrality"):
        if graph.number_of_nodes() == 0:
            st.info("No graph nodes available.")
        else:
            if centrality_choice == "eigenvector":
                centrality = compute_eigenvector_centrality(graph)
            elif centrality_choice == "pagerank":
                centrality = compute_pagerank_centrality(graph, alpha=pagerank_alpha)
            else:
                centrality = compute_degree_centrality(graph)

            top_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
            st.dataframe(
                pd.DataFrame(top_centrality, columns=["node", f"{centrality_choice}_centrality"]),
                use_container_width=True,
            )



def show_outputs_page():
    st.title("Existing outputs")

    root_outputs = sorted(ROOT.glob("*.png")) + sorted(ROOT.glob("*.html")) + sorted(ROOT.glob("*.csv"))
    package_outputs = []
    if OUTPUT_DIR.exists():
        package_outputs = sorted(OUTPUT_DIR.glob("*.png")) + sorted(OUTPUT_DIR.glob("*.html")) + sorted(OUTPUT_DIR.glob("*.csv"))

    all_outputs = root_outputs + package_outputs
    if not all_outputs:
        st.info("No output files found yet.")
        return

    st.write(f"Detected {len(all_outputs)} output files")

    selected = st.selectbox("Open output file", options=all_outputs, format_func=lambda p: str(p.relative_to(ROOT)))

    if selected.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        st.image(str(selected), caption=str(selected.relative_to(ROOT)), use_container_width=True)
    elif selected.suffix.lower() == ".csv":
        out_df = pd.read_csv(selected)
        st.dataframe(out_df.head(200), use_container_width=True)
    elif selected.suffix.lower() == ".html":
        html = selected.read_text(encoding="utf-8", errors="ignore")
        st.components.v1.html(html, height=800, scrolling=True)
    else:
        st.write("Unsupported preview type")


def show_combined_lasso_anchors_page():
    st.title("Combined Lasso + Temporal Anchors")

    selected_lasso_sources = st.multiselect(
        "Datasets to include in combined Lasso",
        options=list(LASSO_COMBINED_SOURCE_MAP.keys()),
        default=list(LASSO_COMBINED_SOURCE_MAP.keys()),
        key="combined_lasso_sources",
    )
    if not selected_lasso_sources:
        st.warning("Select at least one dataset for combined Lasso generation.")
        return

    st.caption(
        "Combined dataset includes: " + ", ".join(selected_lasso_sources) +
        " (basis vectors handled by Lasso graph builder)."
    )

    edge_cutoff = st.slider("Minimum edge weight", 0.0, 1.0, 0.001, 0.001)
    lasso_alpha = st.slider("Lambda (Lasso regularisation)", 0.0001, 1.0, 0.01, 0.0001, key="combined_lasso_lambda")
    max_nodes = st.slider("Max nodes to render", 100, 3000, 900, 50)
    _show_rendered_node_selection_explanation(include_forced_nodes=True)

    combined_lasso_generation_params = {
        "page": "Combined Lasso + Anchors",
        "selected_sources": selected_lasso_sources,
        "edge_cutoff": float(edge_cutoff),
        "lasso_alpha": float(lasso_alpha),
        "max_nodes": int(max_nodes),
    }
    _show_graph_hyperparameters("Graph generation hyperparameters", combined_lasso_generation_params)

    generate_lasso_graph = st.button("Generate combined Lasso graph", key="combined_lasso_generate_button")
    lasso_signature = (tuple(sorted(selected_lasso_sources)), float(edge_cutoff), float(lasso_alpha), int(max_nodes))

    if generate_lasso_graph:
        combined_std = load_combined_standardized_selected(tuple(selected_lasso_sources))
        adjacency = build_lasso_adjacency_from_standardized(combined_std, key="combined_rna_histones", lasso_alpha=lasso_alpha)
        if adjacency.empty:
            st.error("Could not load/generate combined modelled datasets for Lasso analysis.")
            return

        adjacency_filtered = adjacency.copy()
        adjacency_filtered[adjacency_filtered < edge_cutoff] = 0.0
        basis_nodes = [basis for basis in ["e1", "e2", "e3", "e4", "e5"] if basis in adjacency_filtered.index]

        graph = graph_from_adjacency_with_forced_nodes(
            adjacency=adjacency_filtered,
            threshold=edge_cutoff,
            max_nodes=max_nodes,
            forced_nodes=basis_nodes,
        )

        st.session_state["combined_lasso_signature"] = lasso_signature
        st.session_state["combined_lasso_graph_data"] = nx.node_link_data(graph)
        st.session_state["combined_lasso_adjacency_filtered"] = adjacency_filtered
        st.session_state["combined_lasso_basis_nodes"] = basis_nodes
        st.success("Combined Lasso graph generated.")

    if st.session_state.get("combined_lasso_signature") != lasso_signature:
        st.info("Adjust parameters and click 'Generate combined Lasso graph' to update.")
        return

    if "combined_lasso_graph_data" not in st.session_state:
        st.info("Click 'Generate combined Lasso graph' to build the graph.")
        return

    graph = nx.node_link_graph(st.session_state["combined_lasso_graph_data"], directed=True)
    adjacency_filtered = st.session_state["combined_lasso_adjacency_filtered"]
    basis_nodes = st.session_state["combined_lasso_basis_nodes"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", graph.number_of_nodes())
    c2.metric("Edges", graph.number_of_edges())
    c3.metric("Density", f"{nx.density(graph):.5f}" if graph.number_of_nodes() > 1 else "0.00000")

    full_bin = adjacency_filtered > 0
    basis_stats = []
    for basis in basis_nodes:
        basis_stats.append(
            {
                "basis": basis,
                "outgoing_edges": int(full_bin.loc[basis, :].sum()),
                "incoming_edges": int(full_bin.loc[:, basis].sum()),
            }
        )
    if basis_stats:
        st.subheader("Basis vector connectivity")
        st.dataframe(pd.DataFrame(basis_stats), use_container_width=True)

    st.subheader("Centrality visualization")
    centrality_choice = st.selectbox(
        "Node size based on",
        ["degree", "eigenvector", "pagerank"],
        key="combined_lasso_centrality"
    )
    
    pagerank_alpha = 0.85
    if centrality_choice == "pagerank":
        pagerank_alpha = st.slider(
            "PageRank alpha (damping factor)",
            0.1, 0.99, 0.85, 0.05,
            key="combined_lasso_pagerank_alpha"
        )

    fig, closest_to_basis = plot_combined_temporal_graph(
        graph, basis_nodes, 
        centrality_type=centrality_choice, 
        pagerank_alpha=pagerank_alpha
    )
    combined_lasso_params = {
        **combined_lasso_generation_params,
        "centrality": centrality_choice,
        "pagerank_alpha": float(pagerank_alpha) if centrality_choice == "pagerank" else None,
    }
    _plotly_chart_with_download(
        fig,
        base_name="combined_lasso_graph",
        button_label="Download combined Lasso visualization + parameters",
        key="download_combined_lasso_plot",
        params=combined_lasso_params,
    )
    
    with st.expander("Show top centrality"):
        if graph.number_of_nodes() == 0:
            st.info("No graph nodes available.")
        else:
            # Compute selected centrality
            if centrality_choice == "eigenvector":
                centrality = compute_eigenvector_centrality(graph)
            elif centrality_choice == "pagerank":
                centrality = compute_pagerank_centrality(graph, alpha=pagerank_alpha)
            else:
                centrality = compute_degree_centrality(graph)
            
            top_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
            st.dataframe(
                pd.DataFrame(top_centrality, columns=["node", f"{centrality_choice}_centrality"]),
                use_container_width=True
            )

    if closest_to_basis:
        st.subheader("Closest nodes to temporal anchors")
        ordered = []
        for basis in ["e1", "e2", "e3", "e4", "e5"]:
            if basis in closest_to_basis:
                ordered.append({"basis": basis, "closest_node": closest_to_basis[basis]})
        st.dataframe(pd.DataFrame(ordered), use_container_width=True)



def show_diffusion_simulator_page():
    st.title("Random Walk Diffusion Simulator")
    
    st.write(
        "Run flexible random walk simulations on graphs with configurable parameters and restart probabilities."
    )
    
    csv_files = discover_csv_files()
    if not csv_files:
        st.error(f"No CSV files found in: {DATA_DIR}")
        return

    # Graph type selection
    st.subheader("Graph Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        method = st.selectbox("Edge weighting method", 
            ["corr_power", "cor_gaussian_abs", "cor_exponential_abs", "lasso_linear_regression"], 
            key="diffusion_method")
    
    with col2:
        graph_topology = st.selectbox("Graph topology",
            ["Single-layer", "Multi-layer", "Shadow multigraph"],
            key="diffusion_graph_type",
            help="Single-layer: Single correlation/Lasso graph\nMulti-layer: Separate graphs per dataset with layer transitions\nShadow: Includes positive and negative correlations")
    
    with col3:
        use_3d_pca = st.checkbox("3D PCA visualization", value=False, key="diffusion_use_3d_pca",
            help="Project nodes to PCA space for 3D visualization (slower but more informative)")
    
    # Dataset and graph selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if method == "lasso_linear_regression":
            lasso_scope = st.selectbox("Lasso datasets", ["All datasets (combined)", "Single dataset"], key="diffusion_lasso_scope")
            lasso_combined_sources = st.multiselect(
                "Combined Lasso include",
                options=list(LASSO_COMBINED_SOURCE_MAP.keys()),
                default=list(LASSO_COMBINED_SOURCE_MAP.keys()),
                key="diffusion_lasso_combined_sources",
            )
            lasso_alpha = st.slider(
                "Lambda (Lasso regularisation)",
                min_value=0.0001,
                max_value=1.0,
                value=0.01,
                step=0.0001,
                key="diffusion_lasso_lambda",
            )
        else:
            selected_dataset = st.selectbox("Select dataset", options=csv_files, format_func=lambda p: p.name, key="diffusion_dataset")
    
    with col2:
        if graph_topology in ["Multi-layer", "Shadow multigraph"]:
            # Allow selecting multiple datasets for multigraphs
            available_datasets = [f.name for f in csv_files]
            selected_datasets = st.multiselect(
                "Datasets for multigraph",
                options=available_datasets,
                default=available_datasets,  # Default to all datasets
                key="diffusion_multigraph_datasets",
                help="Select which datasets to include as layers"
            )
            if not selected_datasets:
                st.warning("Please select at least one dataset for the multigraph")
        else:
            selected_datasets = []  # Empty list for single-layer graphs
    
    with col3:
        pass  # Placeholder for layout

    # Multigraph transition parameters
    alpha = 0.1
    gamma = 0.01
    if graph_topology in ["Multi-layer", "Shadow multigraph"]:
        st.subheader("Multigraph Parameters")
        p1, p2 = st.columns(2)
        with p1:
            alpha = st.slider(
                "Alpha (inter-layer transition)",
                0.0,
                1.0,
                0.1,
                0.01,
                key="diffusion_multigraph_alpha",
                help="Probability of transitioning between layers at each step"
            )
        with p2:
            if graph_topology == "Shadow multigraph":
                gamma = st.slider(
                    "Gamma (negative jump probability)",
                    0.0,
                    1.0,
                    0.01,
                    0.01,
                    key="diffusion_shadow_gamma",
                    help="Probability of jumping through negative-correlation shadow edges"
                )
            else:
                st.info("Gamma is only used for Shadow multigraph")
    
    # Handle dataset selection for Lasso
    if method == "lasso_linear_regression":
        if lasso_scope == "All datasets (combined)":
            if not lasso_combined_sources:
                st.error("Select at least one dataset for combined Lasso.")
                return
            df = load_combined_standardized_selected(tuple(lasso_combined_sources))
            if df.empty:
                st.error("Could not load combined datasets for Lasso analysis.")
                return
            graph_type = "Combined Lasso"
        else:
            # Single dataset Lasso
            selected_dataset = st.selectbox("Select dataset for Lasso", options=csv_files, format_func=lambda p: p.name, key="diffusion_lasso_dataset")
            raw_df = load_dataset(str(selected_dataset))
            df = standardize_rows(raw_df)
            graph_type = f"Lasso ({selected_dataset.name})"
    else:
        # Load and process data
        raw_df = load_dataset(str(selected_dataset))
        df = standardize_rows(raw_df)
        graph_type = "Correlation-based"
    
    st.subheader("Graph Construction")
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize edge function names for non-multigraph cases
    edge_fn_intra_name = None
    edge_fn_inter_name = None
    edge_fn_negative_name = None
    
    with col1:
        if method == "lasso_linear_regression":
            st.info("Lasso: fixed")
            n_power = 20
        else:
            n_power = st.slider("Power n", 2, 40, 20, key="diffusion_power", help="Exponent for correlation-based edge weights")
    
    with col2:
        if method == "lasso_linear_regression":
            st.info("")
            sigma = 0.05
        else:
            sigma = st.slider("Sigma", 0.001, 1.0, 0.05, 0.001, key="diffusion_sigma", help="Temperature parameter for exponential/gaussian edge weights")
    
    with col3:
        if graph_topology in ["Multi-layer", "Shadow multigraph"]:
            threshold = st.slider(
                "Edge threshold (multigraph)",
                0.0,
                0.05,
                0.0,
                0.001,
                key="diffusion_threshold_multigraph",
                help="Multigraph edges are probability-scaled; keep threshold low to preserve alpha/gamma effects"
            )
        else:
            threshold = st.slider("Edge threshold", 0.0, 1.0, 0.15, 0.01, key="diffusion_threshold")
    
    with col4:
        max_nodes = st.slider("Max nodes", 50, 3000, 300, 50, key="diffusion_max_nodes")
    _show_rendered_node_selection_explanation(include_forced_nodes=False)
    
    # Edge function selection for multigraph/shadow networks
    if graph_topology in ["Multi-layer", "Shadow multigraph"]:
        st.subheader("Edge Function Configuration")
        col1, col2, col3 = st.columns(3)
        
        edge_intra_options = [
            "cor_exponential_shifted",
            "cor_gaussian_shifted",
            "corr_power",
            "cor_exponential_abs",
            "cor_gaussian_abs",
            "intra_layer_corr_exponential_shifted",
            "intra_layer_corr_gaussian_shifted",
        ]
        edge_inter_options = [
            "intra_layer_corr_exponential_shifted",
            "inter_layer_corr_power",
            "cor_exponential_abs_inter",
            "intra_layer_corr_gaussian_shifted",
        ]
        edge_negative_options = [
            "negative_correlation_weight_exponential",
            "negative_correlation_weight",
        ]
        
        with col1:
            edge_fn_intra_name = st.selectbox(
                "Intra-layer edge function",
                edge_intra_options,
                index=0,
                key="diffusion_edge_intra",
                help="Function to compute edge weights within each layer"
            )
        
        with col2:
            edge_fn_inter_name = st.selectbox(
                "Inter-layer edge function",
                edge_inter_options,
                index=1,
                key="diffusion_edge_inter",
                help="Function to compute edge weights between layers"
            )
        
        with col3:
            if graph_topology == "Shadow multigraph":
                edge_fn_negative_name = st.selectbox(
                    "Negative edge function",
                    edge_negative_options,
                    index=0,
                    key="diffusion_edge_negative",
                    help="Function to weight negative correlations in shadow network"
                )
        
        # Edge weight parameters
        st.write("**Edge weight parameters:**")
        col1, col2 = st.columns(2)
        with col1:
            n_power_edge = st.slider("Power n (for edge weights)", 2, 40, n_power, key="diffusion_edge_power", 
                help="Exponent parameter for edge weight functions")
            n_power = n_power_edge
        
        with col2:
            sigma_edge = st.slider("Sigma (for edge weights)", 0.001, 1.0, sigma, 0.001, key="diffusion_edge_sigma",
                help="Temperature parameter for exponential/gaussian edge weights")
            sigma = sigma_edge
    
    # Build adjacency and graph
    if method == "lasso_linear_regression":
        # Use Lasso linear regression
        from DiffusionRWR_model_package.graph_generation.generate_graph_internal import lasso_single_graph
        
        # Get gene names (exclude basis vectors)
        gene_names = [name for name in df.index if not (name.startswith('e') and len(name) == 2)]
        
        if len(gene_names) >= 2:
            start_gene = gene_names[0]
            end_gene = gene_names[-1]
        else:
            start_gene = df.index[0]
            end_gene = df.index[-1]
        
        adjacency_dict = lasso_single_graph(
            {"data": df},
            edge_fn=None,
            start=start_gene,
            end=end_gene,
            return_signs=False,
            lasso_alpha=lasso_alpha,
        )
        adjacency = adjacency_dict["data"]
        # Take absolute values to handle negative Lasso coefficients
        adjacency = adjacency.abs()
    else:
        # Use correlation-based methods
        adjacency = build_adjacency(df, method=method, n_power=n_power if method == "corr_power" else 20, 
                                   sigma=sigma if method != "corr_power" else 0.05)
    
    # Handle multi-graph topologies
    if graph_topology == "Multi-layer":
        st.info("Using multi-layer graph with layer transitions")
        try:
            from DiffusionRWR_model_package.graph_generation.generate_multi_graph import create_multigraph_with_layer_transitions
            from DiffusionRWR_model_package.graph_generation import edge_weight_functions
            
            # Load selected datasets only
            data_dict = {}
            for csv_file in csv_files:
                if csv_file.name in selected_datasets:  # Only include selected datasets
                    dataset_name = csv_file.stem  # filename without extension
                    raw_df = load_dataset(str(csv_file))
                    data_dict[dataset_name] = standardize_rows(raw_df)
            
            if not data_dict:
                st.error("No datasets selected for multigraph. Please select at least one dataset.")
                return
            
            # Get selected edge functions
            edge_fn_intra = getattr(edge_weight_functions, edge_fn_intra_name, None)
            edge_fn_inter = getattr(edge_weight_functions, edge_fn_inter_name, None)
            
            if not edge_fn_intra or not edge_fn_inter:
                st.error(f"Edge functions not found. Intra: {edge_fn_intra_name}, Inter: {edge_fn_inter_name}")
                return
            
            # Build intra-layer graphs for each dataset using selected edge function
            intra_layer_graphs = {}
            for dataset_name, std_df in data_dict.items():
                if method == "corr_power":
                    layer_adjacency = build_adjacency(std_df, method=method, n_power=n_power, sigma=sigma)
                else:
                    layer_adjacency = build_adjacency(std_df, method=method, n_power=20, sigma=sigma)
                
                intra_layer_graphs[dataset_name] = layer_adjacency
            
            # Create multi-layer graph
            multigraph_adjacency = create_multigraph_with_layer_transitions(
                data_dict, 
                intra_layer_graphs,
                edge_fn_inter,
                alpha=alpha,
                start="e1",
                end="e5"
            )
            graph = graph_from_adjacency(multigraph_adjacency, threshold=threshold, max_nodes=max_nodes)
            is_multigraph = True
        except Exception as e:
            st.warning(f"Multi-layer graph creation failed, using single-layer: {str(e)[:100]}")
            graph = graph_from_adjacency(adjacency, threshold=threshold, max_nodes=max_nodes)
            is_multigraph = False
    
    elif graph_topology == "Shadow multigraph":
        st.info("Using shadow multigraph (positive + negative correlations)")
        try:
            from DiffusionRWR_model_package.graph_generation.generate_multi_graph import create_shadow_network_multigraph
            from DiffusionRWR_model_package.graph_generation import edge_weight_functions
            
            # Load selected datasets only
            data_dict = {}
            for csv_file in csv_files:
                if csv_file.name in selected_datasets:  # Only include selected datasets
                    dataset_name = csv_file.stem
                    raw_df = load_dataset(str(csv_file))
                    data_dict[dataset_name] = standardize_rows(raw_df)
            
            if not data_dict:
                st.error("No datasets selected for multigraph. Please select at least one dataset.")
                return
            
            # Get selected edge functions
            edge_fn_intra = getattr(edge_weight_functions, edge_fn_intra_name, None)
            edge_fn_inter = getattr(edge_weight_functions, edge_fn_inter_name, None)
            edge_fn_negative = getattr(edge_weight_functions, edge_fn_negative_name, None)
            
            if not edge_fn_intra or not edge_fn_inter or not edge_fn_negative:
                st.error(f"Edge functions not found.")
                return
            
            # Build intra-layer graphs and sign matrices for each dataset
            intra_graphs = {}
            sign_matrices = {}
            for dataset_name, std_df in data_dict.items():
                # Compute correlation matrix for sign information
                corr_matrix = std_df.T.corr()
                
                if method == "corr_power":
                    layer_adjacency = build_adjacency(std_df, method=method, n_power=n_power, sigma=sigma)
                else:
                    layer_adjacency = build_adjacency(std_df, method=method, n_power=20, sigma=sigma)
                
                # Pass the adjacency matrix DataFrame, not the graph
                intra_graphs[dataset_name] = layer_adjacency
                sign_matrix_values = np.sign(corr_matrix.values)
                sign_matrices[dataset_name] = pd.DataFrame(sign_matrix_values, index=corr_matrix.index, columns=corr_matrix.columns).fillna(0)
            
            shadow_adjacency = create_shadow_network_multigraph(
                data_dict,
                intra_graphs,
                sign_matrices,
                edge_fn_inter,
                gamma=gamma,
                alpha=alpha,
                start="e1",
                end="e5"
            )
            graph = graph_from_adjacency(shadow_adjacency, threshold=threshold, max_nodes=max_nodes)
            is_multigraph = True
        except Exception as e:
            st.warning(f"Shadow multigraph creation failed, using single-layer: {str(e)[:100]}")
            graph = graph_from_adjacency(adjacency, threshold=threshold, max_nodes=max_nodes)
            is_multigraph = False
    
    else:
        # Single-layer graph
        graph = graph_from_adjacency(adjacency, threshold=threshold, max_nodes=max_nodes)
        is_multigraph = False

    
    if graph.number_of_nodes() == 0:
        st.error("No graph nodes after thresholding. Adjust parameters.")
        return
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Graph Nodes", graph.number_of_nodes())
    c2.metric("Graph Edges", graph.number_of_edges())
    c3.metric("Density", f"{nx.density(graph):.4f}")
    
    # Diffusion walk parameters
    st.subheader("Random Walk Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        restart_prob = st.slider("Restart probability", 0.0, 1.0, 0.1, 0.05, key="diffusion_restart")
    
    with col2:
        n_walks = st.number_input("Number of walks", 10, 10000, 100, 50, key="diffusion_n_walks")
    
    with col3:
        max_steps = st.number_input("Max steps per walk", 10, 1000, 100, 10, key="diffusion_max_steps")
    
    with col4:
        start_node = st.selectbox("Start node", sorted(graph.nodes()), key="diffusion_start")

    selected_dataset_name = selected_dataset.name if "selected_dataset" in locals() else None
    diffusion_graph_generation_params = {
        "method": method,
        "graph_topology": graph_topology,
        "dataset": selected_dataset_name,
        "selected_datasets": selected_datasets,
        "lasso_scope": lasso_scope if method == "lasso_linear_regression" else None,
        "lasso_alpha": float(lasso_alpha) if method == "lasso_linear_regression" else None,
        "lasso_combined_sources": lasso_combined_sources if method == "lasso_linear_regression" and lasso_scope == "All datasets (combined)" else None,
        "n_power": int(n_power),
        "sigma": float(sigma),
        "threshold": float(threshold),
        "max_nodes": int(max_nodes),
        "alpha": float(alpha) if graph_topology in ["Multi-layer", "Shadow multigraph"] else None,
        "gamma": float(gamma) if graph_topology == "Shadow multigraph" else None,
        "edge_fn_intra": edge_fn_intra_name,
        "edge_fn_inter": edge_fn_inter_name,
        "edge_fn_negative": edge_fn_negative_name,
    }
    _show_graph_hyperparameters("Graph generation hyperparameters", diffusion_graph_generation_params)

    diffusion_params = {
        "page": "Diffusion Simulator",
        "graph": diffusion_graph_generation_params,
        "diffusion": {
            "restart_prob": float(restart_prob),
            "n_walks": int(n_walks),
            "max_steps": int(max_steps),
            "start_node": start_node,
            "use_3d_pca": bool(use_3d_pca),
        },
    }
    
    # Run simulation button
    if st.button("Run Random Walk Simulation", key="run_diffusion"):
        try:
            with st.spinner("Running random walk simulation..."):
                node_list, visit_counts, simulation_backend = simulate_diffusion_visits(
                    graph=graph,
                    start_node=start_node,
                    n_walks=int(n_walks),
                    max_steps=int(max_steps),
                    restart_prob=float(restart_prob),
                    use_unweighted_transition=(method == "lasso_linear_regression"),
                    prefer_numba=True,
                )
                st.info(simulation_backend)
                
                # Display results
                st.success("Simulation completed!")
                
                # Create a centrality dict for visualization
                visit_centrality = {node: visits for node, visits in zip(node_list, visit_counts)}
                
                # Visualize graph with visit frequencies
                st.subheader("Diffusion Centrality Visualization")
                
                if graph.number_of_nodes() > 0:
                    pos = nx.spring_layout(graph, seed=42, k=0.5)
                    
                    edge_x = []
                    edge_y = []
                    for source, target in graph.edges():
                        x0, y0 = pos[source]
                        x1, y1 = pos[target]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x,
                        y=edge_y,
                        mode="lines",
                        line=dict(width=0.6, color="rgba(140,140,140,0.45)"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                    
                    node_x = []
                    node_y = []
                    node_size = []
                    node_color = []
                    node_text = []
                    
                    start_x = []
                    start_y = []
                    start_text = []
                    
                    # Get visits for non-start nodes only for scaling
                    non_start_visits = [visits for node, visits in zip(node_list, visit_counts) if node != start_node]
                    max_visits = max(non_start_visits) if non_start_visits else 1.0
                    
                    for node in graph.nodes():
                        x, y = pos[node]
                        visits = visit_centrality[node]
                        
                        if node == start_node:
                            # Start node gets fixed properties
                            start_x.append(x)
                            start_y.append(y)
                            start_text.append(f"{node} (START)<br>Visits: {visits:.4f}")
                        else:
                            # Other nodes scale based on non-start node visits
                            node_x.append(x)
                            node_y.append(y)
                            
                            normalized_visits = np.sqrt(visits / max_visits)
                            size = 6 + 25 * normalized_visits
                            
                            node_size.append(size)
                            node_color.append(visits)
                            node_text.append(f"{node}<br>Visits: {visits:.4f}")
                    
                    # Create trace for regular nodes with color scaling
                    node_trace = go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers",
                        marker=dict(
                            size=node_size,
                            color=node_color,
                            symbol="circle",
                            colorscale="Viridis",
                            colorbar=dict(title="Visit Freq"),
                            opacity=0.8,
                            line=dict(width=1, color="white")
                        ),
                        text=node_text,
                        hoverinfo="text",
                        showlegend=False,
                        name="Nodes"
                    )
                    
                    # Create separate trace for start node with fixed red triangle
                    start_node_trace = go.Scatter(
                        x=start_x,
                        y=start_y,
                        mode="markers",
                        marker=dict(
                            size=16,
                            color="red",
                            symbol="triangle-up",
                            opacity=1.0,
                            line=dict(width=2, color="darkred")
                        ),
                        text=start_text,
                        hoverinfo="text",
                        showlegend=False,
                        name="Start Node"
                    )
                    
                    fig_graph = go.Figure(data=[edge_trace, node_trace, start_node_trace])
                    fig_graph.update_layout(
                        title=f"Random Walk Diffusion Centrality ({graph.number_of_nodes()} nodes)",
                        margin=dict(l=10, r=10, t=50, b=10),
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        height=700,
                    )
                    st.plotly_chart(fig_graph, use_container_width=True)
                    graph_zip = _build_export_zip(
                        base_name="diffusion_centrality",
                        params=diffusion_params,
                        plotly_fig=fig_graph,
                    )
                    st.download_button(
                        "Download centrality visualization + parameters",
                        data=graph_zip,
                        file_name="diffusion_centrality_export.zip",
                        mime="application/zip",
                        key="download_diffusion_centrality",
                    )
                
                # 3D PCA Visualization
                if use_3d_pca:
                    st.subheader("3D PCA Visualization of Diffusion")
                    try:
                        from sklearn.decomposition import PCA
                        
                        # Load all data for PCA
                        all_data_list = []
                        all_nodes = []
                        standardized_data = {}
                        common_columns = None
                        
                        for csv_file in csv_files:
                            raw_df = load_dataset(str(csv_file))
                            std_df = standardize_rows(raw_df)
                            standardized_data[csv_file.stem] = std_df
                            if common_columns is None:
                                common_columns = std_df.columns
                            else:
                                common_columns = common_columns.intersection(std_df.columns)

                        if common_columns is None or len(common_columns) == 0:
                            st.warning("3D PCA visualization failed: no common columns across selected datasets")
                            return

                        for dataset_name, std_df in standardized_data.items():
                            aligned_df = std_df.loc[:, common_columns]
                            all_data_list.append(aligned_df.values)
                            all_nodes.extend([f"{dataset_name}:{node}" for node in aligned_df.index])
                        
                        if all_data_list:
                            all_data = np.vstack(all_data_list)
                            
                            # Compute PCA
                            pca = PCA(n_components=3)
                            pca_coords = pca.fit_transform(all_data)
                            
                            # Map visit frequencies to PCA coordinates
                            visit_dict = {node: visits for node, visits in zip(node_list, visit_counts)}

                            def _candidate_node_keys(dataset_name: str, base_node: str) -> list[str]:
                                layered_node = f"{base_node}_{dataset_name}" if dataset_name else base_node
                                return [layered_node, base_node, f"{layered_node}_shadow"]

                            def _is_start_for_point(dataset_name: str, base_node: str) -> bool:
                                return start_node in _candidate_node_keys(dataset_name, base_node)

                            def _lookup_visits(dataset_name: str, base_node: str) -> float:
                                for key in _candidate_node_keys(dataset_name, base_node):
                                    if key in visit_dict:
                                        return visit_dict[key]
                                return 0.0
                            
                            # Create 3D scatter plot (start node separated, fixed style)
                            regular_x, regular_y, regular_z = [], [], []
                            regular_colors, regular_sizes, regular_hover = [], [], []
                            start_x, start_y, start_z = [], [], []
                            start_hover = []

                            non_start_visits_3d = []
                            for node in all_nodes:
                                dataset_name, base_node = node.split(":", 1) if ":" in node else ("", node)
                                if not _is_start_for_point(dataset_name, base_node):
                                    non_start_visits_3d.append(_lookup_visits(dataset_name, base_node))
                            max_visits_3d = max(non_start_visits_3d) if non_start_visits_3d else 1.0

                            for i, node in enumerate(all_nodes):
                                dataset_name, base_node = node.split(":", 1) if ":" in node else ("", node)
                                visits = _lookup_visits(dataset_name, base_node)

                                if _is_start_for_point(dataset_name, base_node):
                                    start_x.append(pca_coords[i, 0])
                                    start_y.append(pca_coords[i, 1])
                                    start_z.append(pca_coords[i, 2])
                                    start_hover.append(f"{node}<br>Visits: {visits:.4f}<br>(START)")
                                else:
                                    regular_x.append(pca_coords[i, 0])
                                    regular_y.append(pca_coords[i, 1])
                                    regular_z.append(pca_coords[i, 2])
                                    regular_colors.append(visits)
                                    normalized_visits = np.sqrt(visits / max_visits_3d) if max_visits_3d > 0 else 0
                                    regular_sizes.append(10 + 34 * normalized_visits)
                                    regular_hover.append(f"{node}<br>Visits: {visits:.4f}")

                            regular_trace = go.Scatter3d(
                                x=regular_x,
                                y=regular_y,
                                z=regular_z,
                                mode='markers',
                                marker=dict(
                                    size=regular_sizes,
                                    color=regular_colors,
                                    colorscale='Viridis',
                                    opacity=0.8,
                                    colorbar=dict(title="Visit Freq"),
                                    line=dict(color='white', width=0.5),
                                    symbol='circle'
                                ),
                                text=regular_hover,
                                hoverinfo='text',
                                name='Nodes'
                            )

                            start_trace = go.Scatter3d(
                                x=start_x,
                                y=start_y,
                                z=start_z,
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color='red',
                                    symbol='diamond',
                                    opacity=1.0,
                                    line=dict(color='darkred', width=2)
                                ),
                                text=start_hover,
                                hoverinfo='text',
                                name='Start Node'
                            )

                            fig_3d = go.Figure(data=[regular_trace, start_trace])
                            
                            fig_3d.update_layout(
                                title=f"3D PCA: Node Distribution and Visit Frequencies<br>Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}, PC3={pca.explained_variance_ratio_[2]:.1%}",
                                scene=dict(
                                    xaxis_title='PC1',
                                    yaxis_title='PC2',
                                    zaxis_title='PC3',
                                    aspectmode='cube',
                                    camera=dict(eye=dict(x=1.25, y=1.25, z=0.95))
                                ),
                                height=950,
                            )
                            st.plotly_chart(fig_3d, use_container_width=True)
                            pca_zip = _build_export_zip(
                                base_name="diffusion_pca_3d",
                                params={
                                    **diffusion_params,
                                    "pca": {
                                        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                                        "n_components": 3,
                                    },
                                },
                                plotly_fig=fig_3d,
                            )
                            st.download_button(
                                "Download 3D PCA visualization + parameters",
                                data=pca_zip,
                                file_name="diffusion_pca_3d_export.zip",
                                mime="application/zip",
                                key="download_diffusion_pca3d",
                            )
                    
                    except Exception as e:
                        st.warning(f"3D PCA visualization failed: {str(e)[:100]}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Visit frequency bar chart (excluding start node)
                    non_start_data = [(node, freq) for node, freq in zip(node_list, visit_counts) if node != start_node]
                    visit_df = pd.DataFrame({
                        'node': [item[0] for item in non_start_data],
                        'visit_frequency': [item[1] for item in non_start_data]
                    }).sort_values('visit_frequency', ascending=False).head(20)
                    
                    fig_visits = px.bar(visit_df, x='visit_frequency', y='node', orientation='h', 
                                       title='Top 20 Most Visited Nodes (excluding start)')
                    st.plotly_chart(fig_visits, use_container_width=True)
                    visits_zip = _build_export_zip(
                        base_name="diffusion_top_visits",
                        params=diffusion_params,
                        plotly_fig=fig_visits,
                        csv_frames={"top_20_visits": visit_df},
                    )
                    st.download_button(
                        "Download visit-frequency chart + parameters",
                        data=visits_zip,
                        file_name="diffusion_top_visits_export.zip",
                        mime="application/zip",
                        key="download_diffusion_visits",
                    )
                
                with col2:
                    # Statistics
                    st.metric("Most visited node", visit_df.iloc[0]['node'])
                    st.metric("Max visit frequency", f"{visit_df.iloc[0]['visit_frequency']:.4f}")
                    st.metric("Mean visit frequency", f"{visit_counts.mean():.4f}")
                    st.metric("Std dev visit frequency", f"{visit_counts.std():.4f}")
                
                # Full results table (excluding start node)
                with st.expander("View all visit frequencies"):
                    full_df = pd.DataFrame({
                        'node': [node for node in node_list if node != start_node],
                        'visit_frequency': [visits for node, visits in zip(node_list, visit_counts) if node != start_node]
                    }).sort_values('visit_frequency', ascending=False)
                    st.dataframe(full_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during simulation: {str(e)}")


def show_start_to_end_page():
    st.title("Start-to-End Diffusion Pipeline")
    st.write("Run the package-level start-to-end diffusion pipeline and review generated outputs.")

    from DiffusionRWR_model_package.diffusion_functions import start_to_end as start_to_end_module

    st.subheader("Hyperparameters")
    c1, c2, c3 = st.columns(3)

    with c1:
        use_fast_rwr = st.checkbox("USE_FAST_RWR", value=bool(start_to_end_module.USE_FAST_RWR), key="s2e_use_fast")
        use_shadow_network = st.checkbox("USE_SHADOW_NETWORK", value=bool(start_to_end_module.USE_SHADOW_NETWORK), key="s2e_use_shadow")
        fix_transition_prob = st.checkbox("fix_transition_prob", value=bool(start_to_end_module.fix_transition_prob), key="s2e_fix_transition")
        gamma = st.slider("gamma", 0.0, 1.0, float(start_to_end_module.gamma), 0.001, format="%.3f", key="s2e_gamma")
        alpha = st.slider("alpha", 0.0, 1.0, float(start_to_end_module.alpha), 0.001, format="%.3f", key="s2e_alpha")

    with c2:
        restart_prob = st.slider("restart_prob", 0.0, 1.0, float(start_to_end_module.restart_prob), 0.01, key="s2e_restart")
        n_simulations = st.number_input("n_simulations", min_value=1, max_value=200000, value=int(start_to_end_module.n_simulations), step=100, key="s2e_n_sims")
        start_node = st.text_input("start", value=str(start_to_end_module.start), key="s2e_start")
        end_node = st.text_input("end", value=str(start_to_end_module.end), key="s2e_end")

    with c3:
        folder_path = st.text_input("folder_path", value=str(start_to_end_module.folder_path), key="s2e_folder")
        edge_intra_options = [
            "cor_exponential_shifted",
            "cor_gaussian_shifted",
            "corr_power",
            "cor_exponential_abs",
            "cor_gaussian_abs",
            "intra_layer_corr_exponential_shifted",
            "intra_layer_corr_gaussian_shifted",
        ]
        edge_inter_options = [
            "intra_layer_corr_exponential_shifted",
            "inter_layer_corr_power",
            "cor_exponential_abs_inter",
            "intra_layer_corr_gaussian_shifted",
        ]
        edge_negative_options = [
            "negative_correlation_weight_exponential",
            "negative_correlation_weight",
        ]

        current_intra = getattr(start_to_end_module.edge_fn_intra, "__name__", "cor_exponential_shifted")
        current_inter = getattr(start_to_end_module.edge_fn_inter, "__name__", "intra_layer_corr_exponential_shifted")
        current_negative = getattr(start_to_end_module.edge_fn_negative, "__name__", "negative_correlation_weight_exponential")

        edge_fn_intra_name = st.selectbox(
            "edge_fn_intra",
            edge_intra_options,
            index=edge_intra_options.index(current_intra) if current_intra in edge_intra_options else 0,
            key="s2e_edge_intra",
        )
        edge_fn_inter_name = st.selectbox(
            "edge_fn_inter",
            edge_inter_options,
            index=edge_inter_options.index(current_inter) if current_inter in edge_inter_options else 0,
            key="s2e_edge_inter",
        )
        edge_fn_negative_name = st.selectbox(
            "edge_fn_negative",
            edge_negative_options,
            index=edge_negative_options.index(current_negative) if current_negative in edge_negative_options else 0,
            key="s2e_edge_negative",
        )
    
    st.write("**Edge weight parameters:**")
    c1, c2 = st.columns(2)
    with c1:
        n_power_s2e = st.slider("n_power (edge weights)", 2, 40, 20, key="s2e_n_power",
            help="Exponent parameter for edge weight functions")
    with c2:
        sigma_s2e = st.slider("sigma (edge weights)", 0.001, 1.0, 0.05, 0.001, key="s2e_sigma",
            help="Temperature parameter for exponential/gaussian edge weights")

    s2e_graph_topology = "shadow_multigraph" if use_shadow_network else ("multigraph_with_layer_transitions" if fix_transition_prob else "multigraph")
    s2e_graph_generation_params = {
        "graph_topology": s2e_graph_topology,
        "edge_fn_intra": edge_fn_intra_name,
        "edge_fn_inter": edge_fn_inter_name,
        "edge_fn_negative": edge_fn_negative_name,
        "n_power": int(n_power_s2e),
        "sigma": float(sigma_s2e),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "start": start_node,
        "end": end_node,
        "folder_path": folder_path,
    }
    _show_graph_hyperparameters("Graph generation hyperparameters", s2e_graph_generation_params)

    s2e_params = {
        "page": "Start-to-End Pipeline",
        "flags": {
            "USE_FAST_RWR": bool(use_fast_rwr),
            "USE_SHADOW_NETWORK": bool(use_shadow_network),
            "fix_transition_prob": bool(fix_transition_prob),
        },
        "graph": s2e_graph_generation_params,
        "diffusion": {
            "restart_prob": float(restart_prob),
            "n_simulations": int(n_simulations),
            "start": start_node,
            "end": end_node,
            "folder_path": folder_path,
        },
    }

    run_pipeline = st.button("Run start_to_end pipeline", key="run_start_to_end_pipeline")

    if run_pipeline:
        import io
        from contextlib import redirect_stdout

        try:
            start_to_end_module.USE_FAST_RWR = use_fast_rwr
            start_to_end_module.USE_SHADOW_NETWORK = use_shadow_network
            start_to_end_module.fix_transition_prob = fix_transition_prob
            start_to_end_module.gamma = float(gamma)
            start_to_end_module.alpha = float(alpha)
            start_to_end_module.restart_prob = float(restart_prob)
            start_to_end_module.n_simulations = int(n_simulations)
            start_to_end_module.start = start_node
            start_to_end_module.end = end_node
            start_to_end_module.folder_path = folder_path
            start_to_end_module.edge_fn_intra = getattr(start_to_end_module, edge_fn_intra_name)
            start_to_end_module.edge_fn_inter = getattr(start_to_end_module, edge_fn_inter_name)
            start_to_end_module.edge_fn_negative = getattr(start_to_end_module, edge_fn_negative_name)
            start_to_end_module.n_power = int(n_power_s2e)
            start_to_end_module.sigma = float(sigma_s2e)

            stdout_buffer = io.StringIO()
            with st.spinner("Running start_to_end pipeline... this can take a while"):
                with redirect_stdout(stdout_buffer):
                    start_to_end_module.start_to_end()

            st.success("start_to_end pipeline completed")

            logs = stdout_buffer.getvalue()
            if logs:
                with st.expander("Pipeline logs"):
                    st.text(logs)

            generated_html = ROOT / "trajectory_visualization.html"
            if generated_html.exists():
                st.subheader("Generated visualization")
                html = generated_html.read_text(encoding="utf-8", errors="ignore")
                st.components.v1.html(html, height=800, scrolling=True)
                s2e_zip = _build_export_zip(
                    base_name="start_to_end_visualization",
                    params=s2e_params,
                    html_content=html,
                )
                st.download_button(
                    "Download start-to-end visualization + parameters",
                    data=s2e_zip,
                    file_name="start_to_end_visualization_export.zip",
                    mime="application/zip",
                    key="download_start_to_end_bundle",
                )
            else:
                st.info("No trajectory_visualization.html was found after execution.")

        except Exception as e:
            st.error(f"start_to_end failed: {str(e)}")


def show_histone_cluster_analysis_page():
    st.title("Histone Cluster Analysis")
    st.write("Run histone-to-RNA diffusion cluster analysis and inspect ranked RNA endpoints.")

    from DiffusionRWR_model_package.diffusion_functions import histone_cluster_analysis as histone_module

    st.subheader("Hyperparameters")
    c1, c2, c3 = st.columns(3)

    with c1:
        use_fast_rwr = st.checkbox("USE_FAST_RWR", value=bool(histone_module.USE_FAST_RWR), key="hist_use_fast")
        use_shadow_network = st.checkbox("USE_SHADOW_NETWORK", value=bool(histone_module.USE_SHADOW_NETWORK), key="hist_use_shadow")
        fix_transition_prob = st.checkbox("fix_transition_prob", value=bool(histone_module.fix_transition_prob), key="hist_fix_transition")
        gamma = st.slider("gamma", 0.0, 1.0, float(histone_module.gamma), 0.01, key="hist_gamma")
        alpha = st.slider("alpha", 0.0, 1.0, float(histone_module.alpha), 0.0001, key="hist_alpha")

    with c2:
        restart_prob = st.slider("restart_prob", 0.0, 1.0, float(histone_module.restart_prob), 0.01, key="hist_restart")
        n_simulations = st.number_input("n_simulations", min_value=1, max_value=200000, value=int(histone_module.n_simulations), step=100, key="hist_n_sims")
        n_genes = st.number_input("n_genes", min_value=1, max_value=1000, value=int(histone_module.n_genes), step=5, key="hist_n_genes")
        end_node = st.text_input("end", value=str(histone_module.end), key="hist_end")

    with c3:
        folder_path = st.text_input("folder_path", value=str(histone_module.folder_path), key="hist_folder")
        edge_intra_options = [
            "cor_exponential_abs",
            "cor_exponential_shifted",
            "cor_gaussian_abs",
            "cor_gaussian_shifted",
            "corr_power",
        ]
        edge_inter_options = [
            "cor_exponential_abs_inter",
            "cor_gaussian_abs_inter",
            "inter_layer_corr_power",
            "intra_layer_corr_exponential_shifted",
            "intra_layer_corr_gaussian_shifted",
        ]
        edge_negative_options = [
            "negative_correlation_weight_exponential",
            "negative_correlation_weight",
        ]

        current_intra = getattr(histone_module.edge_fn_intra, "__name__", "cor_exponential_abs")
        current_inter = getattr(histone_module.edge_fn_inter, "__name__", "cor_exponential_abs_inter")
        current_negative = getattr(histone_module.edge_fn_negative, "__name__", "negative_correlation_weight_exponential")

        edge_fn_intra_name = st.selectbox(
            "edge_fn_intra",
            edge_intra_options,
            index=edge_intra_options.index(current_intra) if current_intra in edge_intra_options else 0,
            key="hist_edge_intra",
        )
        edge_fn_inter_name = st.selectbox(
            "edge_fn_inter",
            edge_inter_options,
            index=edge_inter_options.index(current_inter) if current_inter in edge_inter_options else 0,
            key="hist_edge_inter",
        )
        edge_fn_negative_name = st.selectbox(
            "edge_fn_negative",
            edge_negative_options,
            index=edge_negative_options.index(current_negative) if current_negative in edge_negative_options else 0,
            key="hist_edge_negative",
        )

    histone_dataset_options = _discover_histone_dataset_options(folder_path)
    default_histone_dataset = str(getattr(histone_module, "histone_dataset_filter", "k20me3")).lower()
    if histone_dataset_options:
        default_histone_index = histone_dataset_options.index(default_histone_dataset) if default_histone_dataset in histone_dataset_options else 0
        selected_histone_dataset = st.selectbox(
            "Histone dataset",
            options=histone_dataset_options,
            index=default_histone_index,
            key="hist_dataset_selector",
        )
    else:
        selected_histone_dataset = st.text_input(
            "Histone dataset",
            value=default_histone_dataset,
            key="hist_dataset_selector",
            help="Token used to select histone files (e.g. k20me3, k27me3, k9me2).",
        )

    start_node_candidates = _discover_histone_start_nodes(folder_path, selected_histone_dataset)
    if start_node_candidates:
        current_start = str(histone_module.start)
        default_start_idx = start_node_candidates.index(current_start) if current_start in start_node_candidates else 0
        start_node = st.selectbox(
            "start",
            options=start_node_candidates,
            index=default_start_idx,
            key="hist_start",
            help="Start nodes are genes from the selected histone dataset.",
        )
    else:
        start_node = st.text_input("start", value=str(histone_module.start), key="hist_start")
    
    st.write("**Edge weight parameters:**")
    c1, c2 = st.columns(2)
    with c1:
        n_power_hist = st.slider("n_power (edge weights)", 2, 40, 20, key="hist_n_power",
            help="Exponent parameter for edge weight functions")
    with c2:
        sigma_hist = st.slider("sigma (edge weights)", 0.001, 1.0, 0.05, 0.001, key="hist_sigma",
            help="Temperature parameter for exponential/gaussian edge weights")

    hist_graph_topology = "shadow_multigraph" if use_shadow_network else ("multigraph_with_layer_transitions" if fix_transition_prob else "multigraph")
    hist_graph_generation_params = {
        "graph_topology": hist_graph_topology,
        "histone_dataset": selected_histone_dataset,
        "edge_fn_intra": edge_fn_intra_name,
        "edge_fn_inter": edge_fn_inter_name,
        "edge_fn_negative": edge_fn_negative_name,
        "n_power": int(n_power_hist),
        "sigma": float(sigma_hist),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "start": start_node,
        "end": end_node,
        "folder_path": folder_path,
    }
    _show_graph_hyperparameters("Graph generation hyperparameters", hist_graph_generation_params)

    hist_params = {
        "page": "Histone Cluster Analysis",
        "flags": {
            "USE_FAST_RWR": bool(use_fast_rwr),
            "USE_SHADOW_NETWORK": bool(use_shadow_network),
            "fix_transition_prob": bool(fix_transition_prob),
        },
        "graph": hist_graph_generation_params,
        "diffusion": {
            "restart_prob": float(restart_prob),
            "n_simulations": int(n_simulations),
            "n_genes": int(n_genes),
            "start": start_node,
            "end": end_node,
            "folder_path": folder_path,
        },
    }

    run_analysis = st.button("Run histone_cluster_analysis", key="run_histone_cluster_analysis")

    if run_analysis:
        import io
        from contextlib import redirect_stdout

        try:
            histone_module.USE_FAST_RWR = use_fast_rwr
            histone_module.USE_SHADOW_NETWORK = use_shadow_network
            histone_module.fix_transition_prob = fix_transition_prob
            histone_module.gamma = float(gamma)
            histone_module.alpha = float(alpha)
            histone_module.restart_prob = float(restart_prob)
            histone_module.n_simulations = int(n_simulations)
            histone_module.n_genes = int(n_genes)
            histone_module.start = start_node
            histone_module.end = end_node
            histone_module.folder_path = folder_path
            histone_module.histone_dataset_filter = selected_histone_dataset
            histone_module.edge_fn_intra = getattr(histone_module, edge_fn_intra_name)
            histone_module.edge_fn_inter = getattr(histone_module, edge_fn_inter_name)
            histone_module.edge_fn_negative = getattr(histone_module, edge_fn_negative_name)
            histone_module.n_power = int(n_power_hist)
            histone_module.sigma = float(sigma_hist)

            stdout_buffer = io.StringIO()
            with st.spinner("Running histone cluster analysis... this can take a while"):
                with redirect_stdout(stdout_buffer):
                    results, rna_endpoint_counts, sorted_rna, fig = histone_module.histone_cluster_analysis()

            st.success("histone_cluster_analysis completed")

            logs = stdout_buffer.getvalue()
            if logs:
                with st.expander("Pipeline logs"):
                    st.text(logs)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total walks", len(results))
            c2.metric("Unique RNA endpoints", len(rna_endpoint_counts))
            c3.metric("Top-ranked genes listed", len(sorted_rna))

            if sorted_rna:
                st.subheader("Top RNA endpoints")
                top_df = pd.DataFrame(sorted_rna, columns=["rna_gene", "normalized_score"]).head(50)
                st.dataframe(top_df, use_container_width=True)
            else:
                top_df = pd.DataFrame(columns=["rna_gene", "normalized_score"])

            st.subheader("PCA visualization")
            st.plotly_chart(fig, use_container_width=True)
            hist_zip = _build_export_zip(
                base_name="histone_cluster_visualization",
                params=hist_params,
                plotly_fig=fig,
                csv_frames={"top_rna_endpoints": top_df},
            )
            st.download_button(
                "Download histone visualization + parameters",
                data=hist_zip,
                file_name="histone_cluster_visualization_export.zip",
                mime="application/zip",
                key="download_histone_bundle",
            )

            csv_path = ROOT / "DiffusionRWR_model_package" / "data" / "rna_endpoint_analysis.csv"
            if csv_path.exists():
                st.subheader("RNA endpoint CSV preview")
                csv_df = pd.read_csv(csv_path)
                st.dataframe(csv_df.head(100), use_container_width=True)

        except Exception as e:
            st.error(f"histone_cluster_analysis failed: {str(e)}")


def _plot_clustered_graph(adjacency: pd.DataFrame, labels_df: pd.DataFrame, title: str) -> go.Figure:
    graph = nx.from_pandas_adjacency(adjacency, create_using=nx.DiGraph)
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title}: No nodes")
        return fig

    pos = nx.spring_layout(graph, seed=42, k=0.55)

    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="rgba(130,130,130,0.4)"),
        hoverinfo="none",
        showlegend=False,
    )

    cluster_series = labels_df["cluster"]
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []

    for node in graph.nodes():
        if node not in cluster_series.index:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(int(cluster_series.loc[node]))
        degree = graph.in_degree(node) + graph.out_degree(node)
        node_size.append(8 + 2.5 * np.sqrt(max(degree, 0)))
        node_text.append(f"{node}<br>cluster={int(cluster_series.loc[node])}<br>degree={degree}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Turbo",
            opacity=0.9,
            colorbar=dict(title="Cluster"),
            line=dict(width=0.7, color="white"),
        ),
        text=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=620,
    )
    return fig


def show_graph_clustering_nmi_page():
    st.title("Multi-Graph Clustering + NMI")
    st.write("Generate multiple graph types, run spectral clustering, and compare clusterings via mutual information.")

    csv_files = discover_csv_files()
    if not csv_files:
        st.error(f"No CSV files found in: {DATA_DIR}")
        return

    graph_options = [
        "Correlation Power",
        "Correlation Gaussian |r|",
        "Correlation Exponential |r|",
        "Lasso (combined)",
        "Lasso (single dataset)",
    ]
    selected_graph_types = st.multiselect(
        "Select graph types to generate",
        options=graph_options,
        default=["Correlation Power", "Correlation Gaussian |r|", "Lasso (combined)"],
        key="cluster_nmi_graph_types",
    )

    if not selected_graph_types:
        st.info("Select at least one graph type to continue.")
        return

    st.subheader("Global clustering parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_clusters = st.slider("n_clusters", 2, 20, 4, key="cluster_nmi_n_clusters")
    with c2:
        assign_labels = st.selectbox(
            "assign_labels",
            ["kmeans", "discretize", "cluster_qr"],
            index=0,
            key="cluster_nmi_assign_labels",
        )
    if assign_labels == "discretize":
        st.info(
            "`discretize` means spectral embedding vectors are converted into hard cluster labels by "
            "discretizing (rotating/thresholding) the continuous embedding, rather than running k-means on it."
        )
    with c3:
        symmetrize = st.selectbox(
            "symmetrize",
            ["max", "mean", "min", "none"],
            index=0,
            key="cluster_nmi_symmetrize",
        )
    with c4:
        nmi_method = st.selectbox(
            "MI method",
            ["normalized", "adjusted", "raw"],
            index=0,
            key="cluster_nmi_method",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        clip_negative = st.checkbox("Clip negative weights", value=True, key="cluster_nmi_clip_negative")
    with c6:
        remove_isolated = st.checkbox("Remove isolated nodes", value=True, key="cluster_nmi_remove_isolated")
    with c7:
        add_self_loops = st.checkbox("Add tiny self-loops", value=False, key="cluster_nmi_add_self_loops")
    with c8:
        min_common_nodes = st.number_input(
            "Min common nodes for MI",
            min_value=2,
            max_value=20000,
            value=25,
            step=1,
            key="cluster_nmi_min_common_nodes",
            help="Pairwise MI/NMI is only computed when two clusterings share at least this many nodes.",
        )

    st.subheader("Graph input datasets")
    input_mode = st.selectbox(
        "Input mode",
        options=["Single dataset", "Multi-dataset (multigraph)"],
        index=0,
        key="cluster_nmi_input_mode",
    )

    selected_dataset = None
    selected_multigraph_dataset_names: list[str] = []
    multigraph_alpha = 0.1
    multigraph_inter_method = "inter_layer_corr_power"
    multigraph_inter_n_power = 20
    multigraph_inter_sigma = 0.05
    multigraph_inter_threshold = 0.0

    if input_mode == "Single dataset":
        selected_dataset = st.selectbox(
            "Dataset for graph generation",
            options=csv_files,
            format_func=lambda p: p.name,
            key="cluster_nmi_dataset",
        )
    else:
        dataset_name_options = [p.name for p in csv_files]
        selected_multigraph_dataset_names = st.multiselect(
            "Datasets for multigraph layers",
            options=dataset_name_options,
            default=dataset_name_options,
            key="cluster_nmi_multigraph_datasets",
        )
        if len(selected_multigraph_dataset_names) < 2:
            st.warning("Select at least two datasets for multigraph clustering.")
            return

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            multigraph_alpha = st.slider(
                "alpha",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                key="cluster_nmi_multigraph_alpha",
            )
        with m2:
            multigraph_inter_method = st.selectbox(
                "Inter-layer method",
                options=[
                    "inter_layer_corr_power",
                    "cor_exponential_abs_inter",
                    "cor_gaussian_abs_inter",
                    "intra_layer_corr_exponential_shifted",
                    "intra_layer_corr_gaussian_shifted",
                ],
                index=0,
                key="cluster_nmi_multigraph_inter_method",
            )
        with m3:
            if multigraph_inter_method == "inter_layer_corr_power":
                multigraph_inter_n_power = st.slider(
                    "inter n_power",
                    min_value=2,
                    max_value=40,
                    value=20,
                    step=1,
                    key="cluster_nmi_multigraph_inter_n_power",
                )
            else:
                multigraph_inter_sigma = st.slider(
                    "inter sigma",
                    min_value=0.001,
                    max_value=1.0,
                    value=0.05,
                    step=0.001,
                    key="cluster_nmi_multigraph_inter_sigma",
                )
        with m4:
            multigraph_inter_threshold = st.slider(
                "inter threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.001,
                key="cluster_nmi_multigraph_inter_threshold",
            )

    graph_params: dict[str, dict] = {}

    for graph_type in selected_graph_types:
        with st.expander(f"Parameters: {graph_type}", expanded=True):
            p1, p2, p3 = st.columns(3)
            with p1:
                threshold = st.slider(
                    "Edge threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    step=0.01,
                    key=f"cluster_nmi_threshold_{graph_type}",
                )
            with p2:
                max_nodes = st.slider(
                    "Max nodes",
                    min_value=50,
                    max_value=3000,
                    value=300,
                    step=50,
                    key=f"cluster_nmi_max_nodes_{graph_type}",
                )

            method = None
            n_power = 20
            sigma = 0.05
            lasso_scope = None
            lasso_dataset = None
            lasso_combined_sources = []
            lasso_alpha = 0.01

            with p3:
                if graph_type == "Correlation Power":
                    method = "corr_power"
                    n_power = st.slider(
                        "n_power",
                        min_value=2,
                        max_value=40,
                        value=20,
                        step=1,
                        key=f"cluster_nmi_n_power_{graph_type}",
                    )
                elif graph_type == "Correlation Gaussian |r|":
                    method = "cor_gaussian_abs"
                    sigma = st.slider(
                        "sigma",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.05,
                        step=0.001,
                        key=f"cluster_nmi_sigma_{graph_type}",
                    )
                elif graph_type == "Correlation Exponential |r|":
                    method = "cor_exponential_abs"
                    sigma = st.slider(
                        "sigma",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.05,
                        step=0.001,
                        key=f"cluster_nmi_sigma_{graph_type}",
                    )
                elif graph_type == "Lasso (combined)":
                    lasso_scope = "combined"
                    lasso_alpha = st.slider(
                        "lambda",
                        min_value=0.0001,
                        max_value=1.0,
                        value=0.01,
                        step=0.0001,
                        key=f"cluster_nmi_lasso_lambda_{graph_type}",
                    )
                    lasso_combined_sources = st.multiselect(
                        "Included datasets",
                        options=list(LASSO_COMBINED_SOURCE_MAP.keys()),
                        default=list(LASSO_COMBINED_SOURCE_MAP.keys()),
                        key=f"cluster_nmi_lasso_combined_sources_{graph_type}",
                    )
                elif graph_type == "Lasso (single dataset)":
                    lasso_scope = "single"
                    lasso_dataset = selected_dataset
                    lasso_alpha = st.slider(
                        "lambda",
                        min_value=0.0001,
                        max_value=1.0,
                        value=0.01,
                        step=0.0001,
                        key=f"cluster_nmi_lasso_lambda_{graph_type}",
                    )

            graph_params[graph_type] = {
                "threshold": float(threshold),
                "max_nodes": int(max_nodes),
                "method": method,
                "n_power": int(n_power),
                "sigma": float(sigma),
                "lasso_scope": lasso_scope,
                "lasso_dataset": lasso_dataset,
                "lasso_combined_sources": lasso_combined_sources,
                "lasso_alpha": float(lasso_alpha),
                "input_mode": input_mode,
                "selected_multigraph_dataset_names": selected_multigraph_dataset_names,
                "multigraph_alpha": float(multigraph_alpha),
                "multigraph_inter_method": multigraph_inter_method,
                "multigraph_inter_n_power": int(multigraph_inter_n_power),
                "multigraph_inter_sigma": float(multigraph_inter_sigma),
                "multigraph_inter_threshold": float(multigraph_inter_threshold),
            }

    _show_rendered_node_selection_explanation(include_forced_nodes=False)
    clustering_generation_params = {
        "page": "Multi-Graph Clustering + NMI",
        "global": {
            "n_clusters": int(n_clusters),
            "assign_labels": assign_labels,
            "symmetrize": symmetrize,
            "nmi_method": nmi_method,
            "clip_negative": bool(clip_negative),
            "remove_isolated": bool(remove_isolated),
            "add_self_loops": bool(add_self_loops),
            "min_common_nodes": int(min_common_nodes),
            "input_mode": input_mode,
            "selected_dataset": str(selected_dataset) if selected_dataset is not None else None,
            "selected_multigraph_dataset_names": selected_multigraph_dataset_names,
        },
        "per_graph": graph_params,
    }
    _show_graph_hyperparameters("Graph generation hyperparameters", clustering_generation_params)

    if st.button("Generate graphs, cluster, and compute MI", key="run_cluster_nmi"):
        generated_graphs: dict[str, pd.DataFrame] = {}
        graph_build_log: list[dict] = []

        try:
            for graph_type in selected_graph_types:
                params = graph_params[graph_type]

                if params.get("input_mode") == "Multi-dataset (multigraph)":
                    selected_names = params.get("selected_multigraph_dataset_names", [])
                    if len(selected_names) < 2:
                        st.warning(f"Skipping {graph_type}: select at least two datasets for multigraph mode.")
                        continue

                    name_to_path = {p.name: p for p in csv_files}
                    std_data_dict: dict[str, pd.DataFrame] = {}
                    intra_layer_graphs: dict[str, pd.DataFrame] = {}

                    if graph_type == "Lasso (combined)":
                        selected_sources = params.get("lasso_combined_sources", [])
                        if len(selected_sources) < 2:
                            st.warning("Skipping Lasso (combined): pick at least two included datasets for multigraph mode.")
                            continue

                        selected_layer_names = set(selected_names)
                        for source_name in selected_sources:
                            source_info = LASSO_COMBINED_SOURCE_MAP.get(source_name)
                            if not source_info:
                                continue
                            filename, suffix = source_info
                            layer_name = filename
                            if layer_name not in selected_layer_names:
                                continue
                            source_path = DATA_DIR / filename
                            if not source_path.exists():
                                continue
                            source_df = _load_modelled_csv(source_path)
                            source_std = standardize_rows(source_df)
                            std_data_dict[layer_name] = source_std
                            intra_layer_graphs[layer_name] = build_lasso_adjacency_from_standardized(
                                source_std,
                                key=layer_name,
                                lasso_alpha=params.get("lasso_alpha", 0.01),
                            )
                    else:
                        for dataset_name in selected_names:
                            dataset_path = name_to_path.get(dataset_name)
                            if dataset_path is None:
                                continue
                            raw_df = load_dataset(str(dataset_path))
                            df_std = standardize_rows(raw_df)
                            std_data_dict[dataset_name] = df_std
                            if graph_type in ["Correlation Power", "Correlation Gaussian |r|", "Correlation Exponential |r|"]:
                                intra_layer_graphs[dataset_name] = build_adjacency(
                                    df_std,
                                    method=params["method"],
                                    n_power=params["n_power"],
                                    sigma=params["sigma"],
                                )
                            else:
                                intra_layer_graphs[dataset_name] = build_lasso_adjacency_from_standardized(
                                    df_std,
                                    key=dataset_name,
                                    lasso_alpha=params.get("lasso_alpha", 0.01),
                                )

                    if len(std_data_dict) < 2 or len(intra_layer_graphs) < 2:
                        st.warning(f"Skipping {graph_type}: insufficient dataset layers after loading.")
                        continue

                    adjacency = build_multigraph_adjacency_for_clustering(
                        std_data_dict=std_data_dict,
                        intra_layer_graphs=intra_layer_graphs,
                        alpha=params.get("multigraph_alpha", 0.1),
                        inter_method=params.get("multigraph_inter_method", "inter_layer_corr_power"),
                        inter_n_power=params.get("multigraph_inter_n_power", 20),
                        inter_sigma=params.get("multigraph_inter_sigma", 0.05),
                        inter_layer_threshold=params.get("multigraph_inter_threshold", 0.0),
                    )
                    if adjacency.empty:
                        st.warning(f"Skipping {graph_type}: multigraph adjacency is empty.")
                        continue
                else:
                    if graph_type in ["Correlation Power", "Correlation Gaussian |r|", "Correlation Exponential |r|"]:
                        raw_df = load_dataset(str(selected_dataset))
                        df_std = standardize_rows(raw_df)
                        adjacency = build_adjacency(
                            df_std,
                            method=params["method"],
                            n_power=params["n_power"],
                            sigma=params["sigma"],
                        )
                    elif graph_type == "Lasso (combined)":
                        selected_sources = params.get("lasso_combined_sources", [])
                        if not selected_sources:
                            st.warning("Skipping Lasso (combined): no datasets selected for combined Lasso.")
                            continue
                        df_std = load_combined_standardized_selected(tuple(selected_sources))
                        if df_std.empty:
                            st.warning("Skipping Lasso (combined): combined dataset unavailable.")
                            continue
                        adjacency = build_lasso_adjacency_from_standardized(
                            df_std,
                            key="combined",
                            lasso_alpha=params.get("lasso_alpha", 0.01),
                        )
                    else:
                        raw_df = load_dataset(str(params["lasso_dataset"]))
                        df_std = standardize_rows(raw_df)
                        adjacency = build_lasso_adjacency_from_standardized(
                            df_std,
                            key="single",
                            lasso_alpha=params.get("lasso_alpha", 0.01),
                        )

                graph_nx = graph_from_adjacency(
                    adjacency,
                    threshold=params["threshold"],
                    max_nodes=params["max_nodes"],
                )

                if graph_nx.number_of_nodes() == 0:
                    st.warning(f"Skipping {graph_type}: no nodes after thresholding.")
                    continue

                adjacency_filtered = nx.to_pandas_adjacency(graph_nx, dtype=float)
                if graph_type == "Lasso (single dataset)" and params.get("input_mode") == "Single dataset":
                    graph_name = f"Lasso (single): {selected_dataset.name}"
                elif params.get("input_mode") == "Multi-dataset (multigraph)":
                    graph_name = f"{graph_type} (multigraph)"
                else:
                    graph_name = graph_type
                generated_graphs[graph_name] = adjacency_filtered

                graph_build_log.append(
                    {
                        "graph": graph_name,
                        "nodes": int(graph_nx.number_of_nodes()),
                        "edges": int(graph_nx.number_of_edges()),
                        "total_weight": float(adjacency_filtered.values.sum()),
                        "threshold": params["threshold"],
                        "max_nodes": params["max_nodes"],
                    }
                )

            if not generated_graphs:
                st.error("No graphs were generated successfully. Adjust thresholds/parameters and try again.")
                return

            eligible_graphs = {k: v for k, v in generated_graphs.items() if v.shape[0] >= n_clusters}
            skipped = [k for k, v in generated_graphs.items() if v.shape[0] < n_clusters]

            if skipped:
                st.warning(
                    "Skipping graphs with fewer nodes than n_clusters: " + ", ".join(skipped)
                )

            if not eligible_graphs:
                st.error("No graph has enough nodes for the selected n_clusters.")
                return

            clustering_results = cluster_multiple_graphs(
                graphs=eligible_graphs,
                n_clusters=n_clusters,
                method="spectral",
                assign_labels=assign_labels,
                symmetrize=symmetrize,
                clip_negative=clip_negative,
                remove_isolated=remove_isolated,
                add_self_loops=add_self_loops,
            )

            st.success("Graph generation and spectral clustering completed.")
            st.subheader("Generated graph summary")
            st.dataframe(pd.DataFrame(graph_build_log), use_container_width=True)
            graph_log_lookup = {entry["graph"]: entry for entry in graph_build_log}

            st.subheader("Clustered graph visualizations")
            for graph_name, result in clustering_results.items():
                labels_df = result["labels"]
                affinity_df = result["affinity"]
                fig = _plot_clustered_graph(
                    adjacency=affinity_df,
                    labels_df=labels_df,
                    title=f"{graph_name} | Spectral clusters",
                )
                graph_slug = _slugify_label(graph_name)
                graph_summary = graph_log_lookup.get(graph_name, {})
                cluster_plot_params = {
                    "page": "Multi-Graph Clustering + NMI",
                    "graph_name": graph_name,
                    "graph_summary": graph_summary,
                    "n_clusters": int(n_clusters),
                    "assign_labels": assign_labels,
                    "symmetrize": symmetrize,
                    "clip_negative": bool(clip_negative),
                    "remove_isolated": bool(remove_isolated),
                    "add_self_loops": bool(add_self_loops),
                    "min_common_nodes": int(min_common_nodes),
                }
                cluster_csv = {
                    f"{graph_slug}_clusters": labels_df.reset_index().rename(columns={"index": "node"})
                }
                _plotly_chart_with_download(
                    fig,
                    base_name=f"{graph_slug}_spectral_clusters",
                    button_label=f"Download {graph_name} clusters + parameters",
                    key=f"download_cluster_plot_{graph_slug}",
                    params=cluster_plot_params,
                    csv_frames=cluster_csv,
                )
                st.dataframe(labels_df.head(100), use_container_width=True)

            graph_names = list(clustering_results.keys())
            overlap_matrix = pd.DataFrame(index=graph_names, columns=graph_names, dtype=int)
            for g_a in graph_names:
                idx_a = clustering_results[g_a]["labels"].index
                for g_b in graph_names:
                    idx_b = clustering_results[g_b]["labels"].index
                    overlap_matrix.loc[g_a, g_b] = int(len(idx_a.intersection(idx_b)))

            st.subheader("Node overlap diagnostics")
            st.dataframe(overlap_matrix, use_container_width=True)
            if len(graph_names) >= 2:
                off_diag = overlap_matrix.values.copy()
                np.fill_diagonal(off_diag, np.nan)
                min_overlap = np.nanmin(off_diag)
                if np.isfinite(min_overlap) and min_overlap < int(min_common_nodes):
                    st.warning(
                        "Some graph pairs share very few nodes. This can produce misleadingly high MI/NMI scores. "
                        "Increase overlap (or lower threshold) and/or raise 'Min common nodes for MI'."
                    )

            if len(clustering_results) >= 2:
                mi_matrix, mi_table = calculate_pairwise_mutual_information(
                    clustering_results,
                    method=nmi_method,
                    label_column="cluster",
                    min_common_nodes=int(min_common_nodes),
                )

                st.subheader(f"{nmi_method.upper()} mutual information matrix")
                fig_mi = px.imshow(
                    mi_matrix,
                    text_auto=".3f",
                    color_continuous_scale="Viridis",
                    aspect="auto",
                    title=f"Pairwise {nmi_method.upper()} between graph clusterings",
                )
                fig_mi.update_layout(height=500)
                mi_params = {
                    "page": "Multi-Graph Clustering + NMI",
                    "method": nmi_method,
                    "min_common_nodes": int(min_common_nodes),
                    "graph_names": graph_names,
                }
                _plotly_chart_with_download(
                    fig_mi,
                    base_name="clustering_mi_matrix",
                    button_label=f"Download {nmi_method.upper()} heatmap + parameters",
                    key="download_cluster_mi_heatmap",
                    params=mi_params,
                    csv_frames={"mi_table": mi_table},
                )
                st.dataframe(mi_table, use_container_width=True)
            else:
                st.info("At least two successfully clustered graphs are required to compute mutual information.")

        except Exception as e:
            st.error(f"Clustering/NMI pipeline failed: {str(e)}")


def main():
    st.set_page_config(page_title="DiffusionRWR Visualiser", layout="wide")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Maths",
            "Dataset Explorer",
            "Graph Builder",
            "Multigraph Builder",
            "Combined Lasso + Anchors",
            "Diffusion Simulator",
            "Multi-Graph Clustering + NMI",
            "Start-to-End Pipeline",
            "Histone Cluster Analysis",
            "Existing Outputs",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.write(f"Data folder: {DATA_DIR}")
    st.sidebar.write(f"Output folder: {OUTPUT_DIR}")

    if page == "Overview":
        show_overview()
    elif page == "Maths":
        show_math_page()
    elif page == "Dataset Explorer":
        show_data_page()
    elif page == "Graph Builder":
        show_graph_builder_page()
    elif page == "Multigraph Builder":
        show_multigraph_builder_page()
    elif page == "Combined Lasso + Anchors":
        show_combined_lasso_anchors_page()
    elif page == "Diffusion Simulator":
        show_diffusion_simulator_page()
    elif page == "Multi-Graph Clustering + NMI":
        show_graph_clustering_nmi_page()
    elif page == "Start-to-End Pipeline":
        show_start_to_end_page()
    elif page == "Histone Cluster Analysis":
        show_histone_cluster_analysis_page()
    elif page == "Existing Outputs":
        show_outputs_page()


if __name__ == "__main__":
    main()
