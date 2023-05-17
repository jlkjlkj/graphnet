"""Class(es) implementing layers to be used in `graphnet` models."""

from typing import Any, Callable, Optional, Sequence, Union

from torch.functional import Tensor
from torch.nn import ReLU, Linear, Sequential
from torch_geometric.nn import EdgeConv, radius_graph
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj
from pytorch_lightning import LightningModule
from torch_scatter import scatter_min


class DynEdgeConv(EdgeConv, LightningModule):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        nb_neighbors: int = 8,
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
        # self.global_variables = global_variables

        # )  # will estimate optimal radius
        self.radius_regressor = Sequential(
            Linear(None, 32), ReLU(), Linear(32, 64), ReLU(), Linear(64, 1)
        )  # will estimate optimal radius

    def forward(
        self, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        # Standard EdgeConv forward pass
        # x = super().forward(x, edge_index)
        # YOU WILL NEED TO WRITE SOMETHING HERE FOR THE MLP
        # Recompute adjacency
        # radii = scatter_min(self.radius_regressor(x), data.batch, dim=0).reshape(-1,1)
        # for i in range(len(data_list)):
        #     data_list[i].edge_index = radius_graph(
        #         x=data_list[i].x[:, 0,1,2], r=radii[i,:].item()
        #     )
        # edge_index = radius_graph(
        #     x=x[:, self.features_subset],
        #     r=r,
        #     batch=batch,
        #     max_num_neighbors=64,
        # ).to(self.device)

        return edge_index
