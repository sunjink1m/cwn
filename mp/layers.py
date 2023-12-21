import torch

from typing import Callable, Optional, List
from torch import Tensor
from mp.cell_mp import CochainMessagePassing, CochainMessagePassingParams
from torch_geometric.nn.inits import reset
from torch.nn import Linear, Sequential, BatchNorm1d as BN, Identity
from data.complex import Cochain
from torch_scatter import scatter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from abc import ABC, abstractmethod
import torch.nn.functional as F

class DummyCochainMessagePassing(CochainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size, down_msg_size, boundary_msg_size=None,
                 use_boundary_msg=False, use_down_msg=True):
        super(DummyCochainMessagePassing, self).__init__(up_msg_size, down_msg_size,
                                                       boundary_msg_size=boundary_msg_size,
                                                       use_boundary_msg=use_boundary_msg,
                                                       use_down_msg=use_down_msg)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        # (num_up_adj, x_feature_dim) + (num_up_adj, up_feat_dim)
        # We assume the feature dim is the same across al levels
        return up_x_j + up_attr

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        # (num_down_adj, x_feature_dim) + (num_down_adj, down_feat_dim)
        # We assume the feature dim is the same across al levels
        return down_x_j + down_attr

    def forward(self, cochain: CochainMessagePassingParams):
        up_out, down_out, boundary_out, _ = self.propagate(cochain.up_index, cochain.down_index,
                                                    cochain.boundary_index, None, x=cochain.x,
                                                    up_attr=cochain.kwargs['up_attr'],
                                                    down_attr=cochain.kwargs['down_attr'],
                                                    boundary_attr=cochain.kwargs['boundary_attr'])
        # down or boundary will be zero if one of them is not used.
        return cochain.x + up_out + down_out + boundary_out


class DummyCellularMessagePassing(torch.nn.Module):
    def __init__(self, input_dim=1, max_dim: int = 2, use_boundary_msg=False, use_down_msg=True):
        super(DummyCellularMessagePassing, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = DummyCochainMessagePassing(input_dim, input_dim, boundary_msg_size=input_dim,
                                          use_boundary_msg=use_boundary_msg, use_down_msg=use_down_msg)
            self.mp_levels.append(mp)
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class CINCochainConv(CochainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 msg_up_nn: Callable, msg_down_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False):
        super(CINCochainConv, self).__init__(up_msg_size, down_msg_size, use_boundary_msg=False)
        self.msg_up_nn = msg_up_nn
        self.msg_down_nn = msg_down_nn
        self.update_nn = update_nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        out_up, out_down, _, _ = self.propagate(cochain.up_index, cochain.down_index,
                                             None, None, x=cochain.x,
                                             up_attr=cochain.kwargs['up_attr'],
                                             down_attr=cochain.kwargs['down_attr'])

        out_up += (1 + self.eps) * cochain.x
        out_down += (1 + self.eps) * cochain.x
        return self.update_nn(out_up + out_down)

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_down_nn)
        reset(self.update_nn)
        self.eps.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if up_attr is not None:
            x = torch.cat([up_x_j, up_attr], dim=-1)
            return self.msg_up_nn(x)
        else:
            return self.msg_up_nn(up_x_j)

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        x = torch.cat([down_x_j, down_attr], dim=-1)
        return self.msg_down_nn(x)


class CINConv(torch.nn.Module):
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 msg_up_nn: Callable, msg_down_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2):
        super(CINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = CINCochainConv(up_msg_size, down_msg_size,
                              msg_up_nn, msg_down_nn, update_nn, eps, train_eps)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class EdgeCINConv(torch.nn.Module):
    """
    CIN convolutional layer which performs cochain message passing only
    _up to_ 1-dimensional cells (edges).
    """
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 v_msg_up_nn: Callable, e_msg_down_nn: Callable, e_msg_up_nn: Callable,
                 v_update_nn: Callable, e_update_nn: Callable, eps: float = 0., train_eps=False):
        super(EdgeCINConv, self).__init__()
        self.max_dim = 1
        self.mp_levels = torch.nn.ModuleList()

        v_mp = CINCochainConv(up_msg_size, down_msg_size,
                            v_msg_up_nn, lambda *args: None, v_update_nn, eps, train_eps)
        e_mp = CINCochainConv(up_msg_size, down_msg_size,
                            e_msg_up_nn, e_msg_down_nn, e_update_nn, eps, train_eps)
        self.mp_levels.extend([v_mp, e_mp])

    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=-1)
    
    
class OrientedConv(CochainMessagePassing):
    def __init__(self, dim: int, up_msg_size: int, down_msg_size: int,
                 update_up_nn: Optional[Callable], update_down_nn: Optional[Callable],
                 update_nn: Optional[Callable], act_fn, orient=True):
        super(OrientedConv, self).__init__(up_msg_size, down_msg_size, use_boundary_msg=False)
        self.dim = dim
        self.update_up_nn = update_up_nn
        self.update_down_nn = update_down_nn
        self.update_nn = update_nn
        self.act_fn = act_fn
        self.orient = orient

    def forward(self, cochain: Cochain):
        assert len(cochain.upper_orient) == cochain.upper_index.size(1)
        assert len(cochain.lower_orient) == cochain.lower_index.size(1)
        assert cochain.upper_index.max() < len(cochain.x)
        assert cochain.lower_index.max() < len(cochain.x)

        out_up, out_down, _, _ = self.propagate(cochain.upper_index, cochain.lower_index, None, None, x=cochain.x,
            up_attr=cochain.upper_orient.view(-1, 1), down_attr=cochain.lower_orient.view(-1, 1))

        out_up = self.update_up_nn(out_up)
        out_down = self.update_down_nn(out_down)
        x = self.update_nn(cochain.x)
        return self.act_fn(x + out_up + out_down)

    def reset_parameters(self):
        reset(self.update_up_nn)
        reset(self.update_down_nn)
        reset(self.update_nn)

    # TODO: As a temporary hack, we pass the orientation through the up and down attributes.
    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if self.orient:
            return up_x_j * up_attr
        return up_x_j

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        if self.orient:
            return down_x_j * down_attr
        return down_x_j


class InitReduceConv(torch.nn.Module):

    def __init__(self, reduce='add'):
        """

        Args:
            reduce (str): Way to aggregate boundaries. Can be "sum, add, mean, min, max"
        """
        super(InitReduceConv, self).__init__()
        self.reduce = reduce

    def forward(self, boundary_x, boundary_index):
        features = boundary_x.index_select(0, boundary_index[0])
        out_size = boundary_index[1, :].max() + 1
        return scatter(features, boundary_index[1], dim=0, dim_size=out_size, reduce=self.reduce)

    
class AbstractEmbedVEWithReduce(torch.nn.Module, ABC):
    
    def __init__(self,
                 v_embed_layer: Callable,
                 e_embed_layer: Optional[Callable],
                 init_reduce: InitReduceConv):
        """

        Args:
            v_embed_layer: Layer to embed the integer features of the vertices
            e_embed_layer: Layer (potentially None) to embed the integer features of the edges.
            init_reduce: Layer to initialise the 2D cell features and potentially the edge features.
        """
        super(AbstractEmbedVEWithReduce, self).__init__()
        self.v_embed_layer = v_embed_layer
        self.e_embed_layer = e_embed_layer
        self.init_reduce = init_reduce
    
    @abstractmethod
    def _prepare_v_inputs(self, v_params):
        pass
    
    @abstractmethod
    def _prepare_e_inputs(self, e_params):
        pass
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert 1 <= len(cochain_params) <= 3
        v_params = cochain_params[0]
        e_params = cochain_params[1] if len(cochain_params) >= 2 else None
        c_params = cochain_params[2] if len(cochain_params) == 3 else None

        vx = self.v_embed_layer(self._prepare_v_inputs(v_params))
        out = [vx]

        if e_params is None:
           assert c_params is None
           return out

        reduced_ex = self.init_reduce(vx, e_params.boundary_index)
        ex = reduced_ex
        if e_params.x is not None:
            ex = self.e_embed_layer(self._prepare_e_inputs(e_params))
            # The output of this should be the same size as the vertex features.
            assert ex.size(1) == vx.size(1)
        out.append(ex)

        if c_params is not None:
            # We divide by two in case this was obtained from node aggregation.
            # The division should not do any harm if this is an aggregation of learned embeddings.
            cx = self.init_reduce(reduced_ex, c_params.boundary_index) / 2.
            out.append(cx)

        return out
    
    def reset_parameters(self):
        reset(self.v_embed_layer)
        reset(self.e_embed_layer)

    
class EmbedVEWithReduce(AbstractEmbedVEWithReduce):

    def __init__(self,
                 v_embed_layer: torch.nn.Embedding,
                 e_embed_layer: Optional[torch.nn.Embedding],
                 init_reduce: InitReduceConv):
        super(EmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)
        
    def _prepare_v_inputs(self, v_params):
        assert v_params.x is not None
        assert v_params.x.dim() == 2
        assert v_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return v_params.x.squeeze(1).to(dtype=torch.long)
    
    def _prepare_e_inputs(self, e_params):
        assert self.e_embed_layer is not None
        assert e_params.x.dim() == 2
        assert e_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return e_params.x.squeeze(1).to(dtype=torch.long)


class OGBEmbedVEWithReduce(AbstractEmbedVEWithReduce):
    
    def __init__(self,
                 v_embed_layer: AtomEncoder,
                 e_embed_layer: Optional[BondEncoder],
                 init_reduce: InitReduceConv):
        super(OGBEmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)

    def _prepare_v_inputs(self, v_params):
        assert v_params.x is not None
        assert v_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return v_params.x.to(dtype=torch.long)
    
    def _prepare_e_inputs(self, e_params):
        assert self.e_embed_layer is not None
        assert e_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return e_params.x.to(dtype=torch.long)


class DenseCINCochainConv(CochainMessagePassing):
    """This is a CIN Cochain layer that operates of upper adjacent cells,
    lower adjacent cells, boundary, and coboundaries.
    
    Based on LessSparseCINCochainConv
    """
    def __init__(self, dim: int,
                 up_msg_size: Optional[int] = None,
                 down_msg_size: Optional[int] = None,
                 boundary_msg_size: Optional[int] = None,
                 coboundary_msg_size: Optional[int] = None,
                 use_up_msg: Optional[bool] = True,
                 use_down_msg: Optional[bool] = True,
                 use_boundary_msg: Optional[bool] = True,
                 use_coboundary_msg: Optional[bool] = True,
                 msg_up_nn: Optional[Callable] = None,
                 msg_down_nn: Optional[Callable] = None,
                 msg_boundaries_nn: Optional[Callable] = None,
                 msg_coboundaries_nn: Optional[Callable] = None,
                 update_up_nn: Optional[Callable] = None,
                 update_down_nn: Optional[Callable] = None,
                 update_boundaries_nn: Optional[Callable] = None,
                 update_coboundaries_nn: Optional[Callable] = None,
                 combine_nn: Optional[Callable] = None,
                 eps: float = 0.,
                 train_eps: bool = False,
                 variant='dense'):
        
        self.use_up_msg = use_up_msg
        self.use_down_msg = use_down_msg
        self.use_boundary_msg = use_boundary_msg
        self.use_coboundary_msg = use_coboundary_msg

        # TODO: add bunch of asserts here so that the nn's and the msg_sizes 
        # we need to use are not None!

        super(DenseCINCochainConv, self).__init__(up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                                                coboundary_msg_size=coboundary_msg_size,
                                                use_up_msg=self.use_up_msg,
                                                use_down_msg=self.use_down_msg,
                                                use_boundary_msg=self.use_boundary_msg,
                                                use_coboundary_msg=self.use_coboundary_msg)
        self.dim = dim

        self.msg_up_nn = msg_up_nn                              if self.use_up_msg else None 
        self.msg_down_nn = msg_down_nn                          if self.use_down_msg else None
        self.msg_boundaries_nn = msg_boundaries_nn              if self.use_boundary_msg else None
        self.msg_coboundaries_nn = msg_coboundaries_nn          if self.use_coboundary_msg else None

        self.update_up_nn = update_up_nn                        if self.use_up_msg else None 
        self.update_down_nn = update_down_nn                    if self.use_down_msg else None
        self.update_boundaries_nn = update_boundaries_nn        if self.use_boundary_msg else None
        self.update_coboundaries_nn = update_coboundaries_nn    if self.use_coboundary_msg else None

        self.combine_nn = combine_nn

        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps])) if self.use_up_msg else None # for upper adjacencies
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps])) if self.use_down_msg else None # for lower adjacencies
            self.eps3 = torch.nn.Parameter(torch.Tensor([eps])) if self.use_boundary_msg else None # for boundaries
            self.eps4 = torch.nn.Parameter(torch.Tensor([eps])) if self.use_coboundary_msg else None # for coboundaries
        else:
            self.register_buffer('eps1', torch.Tensor([eps])) if self.use_up_msg else None 
            self.register_buffer('eps2', torch.Tensor([eps])) if self.use_down_msg else None
            self.register_buffer('eps3', torch.Tensor([eps])) if self.use_boundary_msg else None
            self.register_buffer('eps4', torch.Tensor([eps])) if self.use_coboundary_msg else None
        
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        out_up, out_down, out_boundaries, out_coboundaries = self.propagate(
                                                cochain.up_index, 
                                                cochain.down_index,
                                                cochain.boundary_index, 
                                                cochain.coboundary_index, 
                                                x=cochain.x,
                                                up_attr=cochain.kwargs['up_attr'],
                                                down_attr=cochain.kwargs['down_attr'],
                                                boundary_attr=cochain.kwargs['boundary_attr'],
                                                coboundary_attr=cochain.kwargs['coboundary_attr']
                                                )

        # As in GIN, we can learn an injective update function for each multi-set
        out_up += (1 + self.eps1) * cochain.x if self.use_up_msg else 0.
        out_down += (1 + self.eps2) * cochain.x if self.use_down_msg else 0.
        out_boundaries += (1 + self.eps3) * cochain.x if self.use_boundary_msg else 0.
        out_coboundaries += (1 + self.eps4) * cochain.x if self.use_coboundary_msg else 0.

        out_up = self.update_up_nn(out_up) if self.use_up_msg else None
        out_down = self.update_down_nn(out_down) if self.use_down_msg else None
        out_boundaries = self.update_boundaries_nn(out_boundaries) if self.use_boundary_msg else None
        out_coboundaries = self.update_coboundaries_nn(out_coboundaries) if self.use_coboundary_msg else None

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        outs_list = []
        outs_list += [out_up] if self.use_up_msg else []
        outs_list += [out_down] if self.use_down_msg else []
        outs_list += [out_boundaries] if self.use_boundary_msg else []
        outs_list += [out_coboundaries] if self.use_coboundary_msg else []
        return self.combine_nn(torch.cat(outs_list, dim=-1))

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_down_nn)
        reset(self.msg_boundaries_nn)
        reset(self.msg_coboundaries_nn)

        reset(self.update_up_nn)
        reset(self.update_down_nn)
        reset(self.update_boundaries_nn) 
        reset(self.update_coboundaries_nn)

        reset(self.combine_nn)
        
        self.eps1.data.fill_(self.initial_eps) if self.use_up_msg else None 
        self.eps2.data.fill_(self.initial_eps) if self.use_down_msg else None
        self.eps3.data.fill_(self.initial_eps) if self.use_boundary_msg else None
        self.eps4.data.fill_(self.initial_eps) if self.use_coboundary_msg else None

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        return self.msg_up_nn((up_x_j, up_attr))

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        return self.msg_down_nn((down_x_j, down_attr))
    
    def message_boundary(self, boundary_x_j: Tensor) -> Tensor:
        return self.msg_boundaries_nn(boundary_x_j)
    
    def message_coboundary(self, coboundary_x_j: Tensor) -> Tensor:
        return self.msg_coboundaries_nn(coboundary_x_j)
    
 
class DenseCINConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from cellular upper
    neighbors, cellular lower neighbors, boundaries, and boundaries.
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    def __init__(self, up_msg_size: int, 
                 down_msg_size: int, 
                 boundary_msg_size: Optional[int] = None,
                 coboundary_msg_size: Optional[int] = None,
                 passed_msg_up_nn: Optional[Callable] = None,
                 passed_msg_boundaries_nn: Optional[Callable] = None,
                 passed_msg_down_nn: Optional[Callable] = None,
                 passed_msg_coboundaries_nn: Optional[Callable] = None,
                 passed_update_up_nn: Optional[Callable] = None,
                 passed_update_down_nn: Optional[Callable] = None,
                 passed_update_boundaries_nn: Optional[Callable] = None,
                 passed_update_coboundaries_nn: Optional[Callable] = None,
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_up_attr=False,
                 use_down_attr=False, 
                 omit_2cell_down=False,
                 variant='dense',
                 **kwargs):
        super(DenseCINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()

        if variant == 'gnn': # default graph neural network over original graph
            self.use_up_msg = True
            self.use_down_msg = False
            self.use_boundary_msg = False
            self.use_coboundary_msg = False
        elif variant == 'sparse':
            self.use_up_msg = True
            self.use_down_msg = False
            self.use_boundary_msg = True
            self.use_coboundary_msg = False
        elif variant == 'less-sparse':
            self.use_up_msg = True
            self.use_down_msg = True
            self.use_boundary_msg = True
            self.use_coboundary_msg = False
        elif variant == 'dense':
            self.use_up_msg = True
            self.use_down_msg = True
            self.use_boundary_msg = True
            self.use_coboundary_msg = True

        for dim in range(max_dim+1):
            """
            We sort out some things in order to get rid of unused learnable parameters.
            This means we 
            1. Set use_X_msg to false in certain dimensions (e.g. turn off msg_up 
                in the highest dimension) so that, even though we do pass the 
                appropriate nn's into the CochainConv, they will be ignored.
            2. Set the length of the combine_nn appropriately so we don't need 
                to pass on zero feature vectors in order to match the dimensions.
            """
            bottom_dim = (dim == 0) # there are no down_msg's and boundary links in the bottom dimension
            top_dim = (dim == max_dim) # there are no up_msg's and coboundary links in the bottom dimension

            # if in bottom dim, ignore up_msg and coboundary links (because there are none)
            use_up_msg = self.use_up_msg and (not top_dim)
            use_down_msg = self.use_down_msg and (not bottom_dim)
            use_boundary_msg = self.use_boundary_msg and (not bottom_dim)
            use_coboundary_msg = self.use_coboundary_msg and (not top_dim)
            
            # we can optionally omit 2-cell down adj for regularization
            if top_dim and omit_2cell_down: 
                use_down_msg = False
            
            # count how many types of 'adjacencies' we are using
            num_adjs = use_up_msg + use_down_msg + use_boundary_msg + use_coboundary_msg
            
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_up_attr:
                    msg_up_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_down_nn = passed_msg_down_nn
            if msg_down_nn is None:
                if use_down_attr:
                    msg_down_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_down_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            msg_coboundaries_nn = passed_msg_coboundaries_nn
            if msg_coboundaries_nn is None:
                msg_coboundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_down_nn = passed_update_down_nn
            if update_down_nn is None:
                update_down_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_coboundaries_nn = passed_update_coboundaries_nn
            if update_coboundaries_nn is None:
                update_coboundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
            
            combine_nn = Sequential(
                Linear(kwargs['hidden']*num_adjs, kwargs['hidden']),
                graph_norm(kwargs['hidden']),
                kwargs['act_module']())

            mp = DenseCINCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                coboundary_msg_size=coboundary_msg_size,
                msg_up_nn=msg_up_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                msg_down_nn=msg_down_nn, update_down_nn=update_down_nn,
                msg_coboundaries_nn=msg_coboundaries_nn, update_coboundaries_nn=update_coboundaries_nn,
                update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps, variant=variant,
                use_up_msg = use_up_msg,
                use_down_msg = use_down_msg,
                use_boundary_msg = use_boundary_msg,
                use_coboundary_msg = use_coboundary_msg)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class DenseBasicCochainConv(CochainMessagePassing):
    """This is a CIN Cochain layer that operates of upper adjacent cells,
    lower adjacent cells, boundary, and coboundaries.
    
    Based on LessSparseCINCochainConv
    """
    def __init__(self, dim: int,
                 up_msg_size: Optional[int] = None,
                 down_msg_size: Optional[int] = None,
                 boundary_msg_size: Optional[int] = None,
                 coboundary_msg_size: Optional[int] = None,
                 use_up_msg: Optional[bool] = True,
                 use_down_msg: Optional[bool] = True,
                 use_boundary_msg: Optional[bool] = True,
                 use_coboundary_msg: Optional[bool] = True,
                 msg_up_nn: Optional[Callable] = None,
                 msg_down_nn: Optional[Callable] = None,
                 msg_boundaries_nn: Optional[Callable] = None,
                 msg_coboundaries_nn: Optional[Callable] = None,
                 update_up_nn: Optional[Callable] = None,
                 update_down_nn: Optional[Callable] = None,
                 update_boundaries_nn: Optional[Callable] = None,
                 update_coboundaries_nn: Optional[Callable] = None,
                 combine_nn: Optional[Callable] = None,
                 eps: float = 0.,
                 train_eps: bool = False,
                 variant='dense'):
        
        self.use_up_msg = use_up_msg
        self.use_down_msg = use_down_msg
        self.use_boundary_msg = use_boundary_msg
        self.use_coboundary_msg = use_coboundary_msg

        # TODO: add bunch of asserts here so that the nn's and the msg_sizes 
        # we need to use are not None!

        super(DenseBasicCochainConv, self).__init__(up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                                                coboundary_msg_size=coboundary_msg_size,
                                                use_up_msg=self.use_up_msg,
                                                use_down_msg=self.use_down_msg,
                                                use_boundary_msg=self.use_boundary_msg,
                                                use_coboundary_msg=self.use_coboundary_msg)
        self.dim = dim

        self.msg_up_nn = msg_up_nn                              if self.use_up_msg else None 
        self.msg_down_nn = msg_down_nn                          if self.use_down_msg else None
        self.msg_boundaries_nn = msg_boundaries_nn              if self.use_boundary_msg else None
        self.msg_coboundaries_nn = msg_coboundaries_nn          if self.use_coboundary_msg else None

        self.update_up_nn = update_up_nn                        if self.use_up_msg else None 
        self.update_down_nn = update_down_nn                    if self.use_down_msg else None
        self.update_boundaries_nn = update_boundaries_nn        if self.use_boundary_msg else None
        self.update_coboundaries_nn = update_coboundaries_nn    if self.use_coboundary_msg else None

        self.combine_nn = combine_nn

        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps])) if self.use_up_msg else None # for upper adjacencies
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps])) if self.use_down_msg else None # for lower adjacencies
            self.eps3 = torch.nn.Parameter(torch.Tensor([eps])) if self.use_boundary_msg else None # for boundaries
            self.eps4 = torch.nn.Parameter(torch.Tensor([eps])) if self.use_coboundary_msg else None # for coboundaries
        else:
            self.register_buffer('eps1', torch.Tensor([eps])) if self.use_up_msg else None 
            self.register_buffer('eps2', torch.Tensor([eps])) if self.use_down_msg else None
            self.register_buffer('eps3', torch.Tensor([eps])) if self.use_boundary_msg else None
            self.register_buffer('eps4', torch.Tensor([eps])) if self.use_coboundary_msg else None
        
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        out_up, out_down, out_boundaries, out_coboundaries = self.propagate(
                                                cochain.up_index, 
                                                cochain.down_index,
                                                cochain.boundary_index, 
                                                cochain.coboundary_index, 
                                                x=cochain.x,
                                                up_attr=cochain.kwargs['up_attr'],
                                                down_attr=cochain.kwargs['down_attr'],
                                                boundary_attr=cochain.kwargs['boundary_attr'],
                                                coboundary_attr=cochain.kwargs['coboundary_attr']
                                                )

        # # As in GIN, we can learn an injective update function for each multi-set
        # out_up += (1 + self.eps1) * cochain.x if self.use_up_msg else 0.
        # out_down += (1 + self.eps2) * cochain.x if self.use_down_msg else 0.
        # out_boundaries += (1 + self.eps3) * cochain.x if self.use_boundary_msg else 0.
        # out_coboundaries += (1 + self.eps4) * cochain.x if self.use_coboundary_msg else 0.

        # out_up = self.update_up_nn(out_up) if self.use_up_msg else None
        # out_down = self.update_down_nn(out_down) if self.use_down_msg else None
        # out_boundaries = self.update_boundaries_nn(out_boundaries) if self.use_boundary_msg else None
        # out_coboundaries = self.update_coboundaries_nn(out_coboundaries) if self.use_coboundary_msg else None

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        outs_list = [cochain.x]
        outs_list += [out_up] if self.use_up_msg else []
        outs_list += [out_down] if self.use_down_msg else []
        outs_list += [out_boundaries] if self.use_boundary_msg else []
        outs_list += [out_coboundaries] if self.use_coboundary_msg else []
        return self.combine_nn(torch.cat(outs_list, dim=-1))

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_down_nn)
        reset(self.msg_boundaries_nn)
        reset(self.msg_coboundaries_nn)

        reset(self.update_up_nn)
        reset(self.update_down_nn)
        reset(self.update_boundaries_nn) 
        reset(self.update_coboundaries_nn)

        reset(self.combine_nn)
        
        self.eps1.data.fill_(self.initial_eps) if self.use_up_msg else None 
        self.eps2.data.fill_(self.initial_eps) if self.use_down_msg else None
        self.eps3.data.fill_(self.initial_eps) if self.use_boundary_msg else None
        self.eps4.data.fill_(self.initial_eps) if self.use_coboundary_msg else None

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        return self.msg_up_nn((up_x_j, up_attr))

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        return self.msg_down_nn((down_x_j, down_attr))
    
    def message_boundary(self, boundary_x_j: Tensor) -> Tensor:
        return self.msg_boundaries_nn(boundary_x_j)
    
    def message_coboundary(self, coboundary_x_j: Tensor) -> Tensor:
        return self.msg_coboundaries_nn(coboundary_x_j)
    
 
class DenseBasicConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from cellular upper
    neighbors, cellular lower neighbors, boundaries, and boundaries.
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    def __init__(self, up_msg_size: int, 
                 down_msg_size: int, 
                 boundary_msg_size: Optional[int] = None,
                 coboundary_msg_size: Optional[int] = None,
                 passed_msg_up_nn: Optional[Callable] = None,
                 passed_msg_boundaries_nn: Optional[Callable] = None,
                 passed_msg_down_nn: Optional[Callable] = None,
                 passed_msg_coboundaries_nn: Optional[Callable] = None,
                 passed_update_up_nn: Optional[Callable] = None,
                 passed_update_down_nn: Optional[Callable] = None,
                 passed_update_boundaries_nn: Optional[Callable] = None,
                 passed_update_coboundaries_nn: Optional[Callable] = None,
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_up_attr=False,
                 use_down_attr=False, 
                 omit_2cell_down=False,
                 variant='dense',
                 **kwargs):
        super(DenseBasicConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()

        if variant == 'gnn': # default graph neural network over original graph
            self.use_up_msg = True
            self.use_down_msg = False
            self.use_boundary_msg = False
            self.use_coboundary_msg = False
        elif variant == 'sparse':
            self.use_up_msg = True
            self.use_down_msg = False
            self.use_boundary_msg = True
            self.use_coboundary_msg = False
        elif variant == 'less-sparse':
            self.use_up_msg = True
            self.use_down_msg = True
            self.use_boundary_msg = True
            self.use_coboundary_msg = False
        elif variant == 'dense':
            self.use_up_msg = True
            self.use_down_msg = True
            self.use_boundary_msg = True
            self.use_coboundary_msg = True

        for dim in range(max_dim+1):
            """
            We sort out some things in order to get rid of unused learnable parameters.
            This means we 
            1. Set use_X_msg to false in certain dimensions (e.g. turn off msg_up 
                in the highest dimension) so that, even though we do pass the 
                appropriate nn's into the CochainConv, they will be ignored.
            2. Set the length of the combine_nn appropriately so we don't need 
                to pass on zero feature vectors in order to match the dimensions.
            """
            bottom_dim = (dim == 0) # there are no down_msg's and boundary links in the bottom dimension
            top_dim = (dim == max_dim) # there are no up_msg's and coboundary links in the bottom dimension

            # if in bottom dim, ignore up_msg and coboundary links (because there are none)
            use_up_msg = self.use_up_msg and (not top_dim)
            use_down_msg = self.use_down_msg and (not bottom_dim)
            use_boundary_msg = self.use_boundary_msg and (not bottom_dim)
            use_coboundary_msg = self.use_coboundary_msg and (not top_dim)
            
            # we can optionally omit 2-cell down adj for regularization
            if top_dim and omit_2cell_down: 
                use_down_msg = False
            
            # count how many types of 'adjacencies' we are using
            num_adjs = use_up_msg + use_down_msg + use_boundary_msg + use_coboundary_msg
            
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_up_attr:
                    msg_up_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_down_nn = passed_msg_down_nn
            if msg_down_nn is None:
                if use_down_attr:
                    msg_down_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_down_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            msg_coboundaries_nn = passed_msg_coboundaries_nn
            if msg_coboundaries_nn is None:
                msg_coboundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Identity()

            update_down_nn = passed_update_down_nn
            if update_down_nn is None:
                update_down_nn = Identity()

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Identity()

            update_coboundaries_nn = passed_update_coboundaries_nn
            if update_coboundaries_nn is None:
                update_coboundaries_nn = Identity()
            
            combine_nn = Sequential(
                Linear(kwargs['layer_dim']*(num_adjs+1), kwargs['layer_dim']*2),
                graph_norm(kwargs['layer_dim']*2),
                kwargs['act_module'](),
                Linear(kwargs['layer_dim']*2, kwargs['hidden']), # parametrize this class so it's (l_d*n, l_d*m) here.
                graph_norm(kwargs['hidden']),
                kwargs['act_module'](),
                Linear(kwargs['hidden'], kwargs['hidden']),
                graph_norm(kwargs['hidden']),
                kwargs['act_module']())

            mp = DenseBasicCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                coboundary_msg_size=coboundary_msg_size,
                msg_up_nn=msg_up_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                msg_down_nn=msg_down_nn, update_down_nn=update_down_nn,
                msg_coboundaries_nn=msg_coboundaries_nn, update_coboundaries_nn=update_coboundaries_nn,
                update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps, variant=variant,
                use_up_msg = use_up_msg,
                use_down_msg = use_down_msg,
                use_boundary_msg = use_boundary_msg,
                use_coboundary_msg = use_coboundary_msg)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out

