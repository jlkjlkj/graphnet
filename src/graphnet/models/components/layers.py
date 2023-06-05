"""Class(es) implementing layers to be used in `graphnet` models."""

from typing import Any, Callable, Optional, Sequence, Union

from torch.functional import Tensor
from torch.nn import ReLU, Linear, Sequential
from torch_geometric.nn import EdgeConv, radius_graph
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj
from pytorch_lightning import LightningModule
from torch_scatter import scatter_min
from torch_geometric.data import Data
import torch


class DynEdgeConv(EdgeConv, LightningModule):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        nb_neighbors: int = 8,
        mlp_input: int = 256,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        # global_variables: Optional[Union[Sequence[int], slice]],
        **kwargs: Any,
    ):
        """Construct `DynEdgeConv`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            mlp_input: input size of mlp used for radii approximation.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Base class constructor
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset
        self.mlp_input = mlp_input
        # Will estimate optimal radii
        self.radius_regressor = Sequential(
            Linear(self.mlp_input, 512),
            ReLU(),
            Linear(256, 1),
        )
        # self.radius_regressor = Sequential(
        #     Linear(self.mlp_input, 32),
        #     ReLU(),
        #     Linear(32, 64),
        #     ReLU(),
        #     Linear(64, 1),
        # )

    # def forward(
    #     self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    # ) -> Tensor:
    #     """Forward pass."""
    #     # putting aside x
    #     tmp_x = torch.clone(x[:, 0 : self.mlp_input])
    #     # Standard EdgeConv forward pass
    #     x = super().forward(x, edge_index)
    #     # getting minimum of radius per graph
    #     radii = scatter_min(
    #         self.radius_regressor(tmp_x).view(-1), batch, dim=0
    #     )[0]

    #     # Bucketizing single graphs
    #     batch_pos = torch.bucketize(
    #         torch.arange(torch.unique(batch).size(dim=0) + 1).to(self.device),
    #         batch,
    #     )

    #     # creating graph edge indices for each graph
    #     edge_index_list = []
    #     for idx in range(len(batch_pos) - 1):
    #         graph_x = tmp_x[batch_pos[idx] : batch_pos[idx + 1]]
    #         # bulding edge_index based on indiviudal radius
    #         edge_index = torch.add(
    #             radius_graph(graph_x[:, 0:3], r=radii[idx]), batch_pos[idx]
    #         ).to(self.device)
    #         edge_index_list.append(edge_index)
    #     edge_index = torch.cat(edge_index_list, dim=1)

    #     return x, edge_index
    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        # putting aside x
        # tmp_x = torch.clone(x[:, 0 : self.mlp_input])

        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index)
        # getting minimum of radius per graph
        radii = scatter_min(self.radius_regressor(x).view(-1), batch, dim=0)[0]

        # Bucketizing single graphs
        batch_pos = torch.bucketize(
            torch.arange(torch.unique(batch).size(dim=0) + 1).to(self.device),
            batch,
        )

        # creating graph edge indices for each graph
        edge_index_list = []
        for idx in range(len(batch_pos) - 1):
            graph_x = x[batch_pos[idx] : batch_pos[idx + 1]]
            # bulding edge_index based on indiviudal radius
            edge_index = torch.add(
                radius_graph(graph_x[:, 0:3], r=radii[idx]), batch_pos[idx]
            ).to(self.device)
            edge_index_list.append(edge_index)
        edge_index = torch.cat(edge_index_list, dim=1)

        return x, edge_index
