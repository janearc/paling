import numpy as np
import hdbscan
import umap
from sklearn.neighbors import kneighbors_graph
from typing import List, Dict, Any, Union

class TopologyBanchan:
    """
    A banchan class for Lexical Cartography (Step 1 of Edge Exploration).
    Responsible for ingesting embeddings, projecting, clustering, and 
    building a topology graph for visual rendering.
    """
    def __init__(
        self,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 2,
        umap_metric: str = "cosine",
        hdbscan_min_cluster_size: int = 15,
        hdbscan_min_samples: int = 5,
        knn_n_neighbors: int = 5
    ):
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.knn_n_neighbors = knn_n_neighbors
        
        # Internal state
        self.embeddings = None
        self.projection = None
        self.cluster_labels = None
        self.cluster_probabilities = None
        self.knn_graph = None

    def ingest(self, embeddings: Union[np.ndarray, List[List[float]]]) -> "TopologyBanchan":
        """
        Ingest a dataset of conversational embeddings.
        """
        self.embeddings = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
        return self

    def project(self) -> "TopologyBanchan":
        """
        Project embeddings via UMAP.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to project. Call ingest() first.")
        
        reducer = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=self.umap_n_components,
            metric=self.umap_metric,
            random_state=42
        )
        self.projection = reducer.fit_transform(self.embeddings)
        return self

    def cluster(self) -> "TopologyBanchan":
        """
        Cluster embeddings via HDBSCAN.
        Uses 'eom' (Excess of Mass) for multidimensional definitions of clusters
        with variable density thresholds, rather than a single distance threshold.
        """
        if self.projection is None:
            raise ValueError("No projection to cluster. Call project() first.")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            cluster_selection_method='eom',
            prediction_data=True
        )
        clusterer.fit(self.projection)
        self.cluster_labels = clusterer.labels_
        self.cluster_probabilities = clusterer.probabilities_
        return self

    def connect(self) -> "TopologyBanchan":
        """
        Connect the topology via a scikit-learn KNN graph on the projected space.
        """
        if self.projection is None:
            raise ValueError("No projection to connect. Call project() first.")
        
        self.knn_graph = kneighbors_graph(
            self.projection,
            n_neighbors=self.knn_n_neighbors,
            mode='distance',
            include_self=False
        )
        return self

    def export(self) -> Dict[str, Any]:
        """
        Define the output of this class to be a data structure suitable 
        for a custom visual rendering engine.
        """
        if self.projection is None or self.cluster_labels is None or self.knn_graph is None:
            raise ValueError("Pipeline not fully executed. Run ingest().project().cluster().connect() first.")
        
        nodes = []
        for i in range(len(self.projection)):
            nodes.append({
                "id": i,
                "x": float(self.projection[i, 0]),
                "y": float(self.projection[i, 1]) if self.umap_n_components > 1 else 0.0,
                "cluster": int(self.cluster_labels[i]),
                "probability": float(self.cluster_probabilities[i])
            })
            
        edges = []
        coo = self.knn_graph.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            edges.append({
                "source": int(i),
                "target": int(j),
                "distance": float(v)
            })
            
        num_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "num_clusters": num_clusters
            }
        }
