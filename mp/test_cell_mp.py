import pytest
import torch

from data.helper_test import check_edge_index_are_the_same, check_edge_attr_are_the_same

from mp.cell_mp import CochainMessagePassing
from torch_geometric.nn.conv import MessagePassing
from data.dummy_complexes import (get_square_dot_complex, get_house_complex,
                                  get_colon_complex, get_fullstop_complex, 
                                  get_bridged_complex, convert_to_graph, 
                                  get_molecular_complex)
from data.utils import compute_ring_2complex

@pytest.mark.parametrize("include_coboundary_links", [True, False])
def test_edge_propagate_at_edge_level_in_cmp(include_coboundary_links):
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the edge level."""

    house_complex = get_house_complex(include_coboundary_links)
    e = house_complex.get_cochain_params(dim=1)
    assert e.kwargs['boundary_index'] is not None, e.kwargs['boundary_index']

    # Extract the message passing object and propagate
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1,
                 use_coboundary_msg=include_coboundary_links)
    up_msg, down_msg, boundary_msg, coboundary_msg = cmp.propagate(e.up_index, e.down_index,
                                               e.boundary_index, e.coboundary_index, x=e.x,
                                               up_attr=e.kwargs['up_attr'],
                                               down_attr=e.kwargs['down_attr'],
                                               boundary_attr=e.kwargs['boundary_attr'],
                                               coboundary_attr=e.kwargs['coboundary_attr'])
    expected_down_msg = torch.tensor([[6], [10], [17], [9], [13], [10]], dtype=torch.float)
    assert torch.equal(down_msg, expected_down_msg)

    expected_up_msg = torch.tensor([[0], [0], [11], [0], [9], [8]], dtype=torch.float)
    assert torch.equal(up_msg, expected_up_msg)

    expected_boundary_msg = torch.tensor([[3], [5], [7], [5], [9], [8]], dtype=torch.float)
    assert torch.equal(boundary_msg, expected_boundary_msg)

    if include_coboundary_links:
        expected_coboundary_msg = torch.tensor([[0], [0], [1], [0], [1], [1]], dtype=torch.float)
    else:
        expected_coboundary_msg = torch.tensor([[0], [0], [0], [0], [0], [0]], dtype=torch.float)
    assert torch.equal(coboundary_msg, expected_coboundary_msg)
        

@pytest.mark.parametrize("include_coboundary_links", [True, False])
def test_propagate_at_vertex_level_in_cmp(include_coboundary_links):
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the vertex level. This makes sure propagate works when
    down_index is None.
    """

    house_complex = get_house_complex(include_coboundary_links)
    v = house_complex.get_cochain_params(dim=0)
    ###################### FIX: v.coboundary_attr is None here... It shouldn't be
    # Extract the message passing object and propagate
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1,
                 use_coboundary_msg=include_coboundary_links)
    up_msg, down_msg, boundary_msg, coboundary_msg = cmp.propagate(v.up_index, v.down_index,
                                               v.boundary_index, v.coboundary_index, x=v.x,
                                               up_attr=v.kwargs['up_attr'],
                                               down_attr=v.kwargs['down_attr'],
                                               boundary_attr=v.kwargs['boundary_attr'],
                                               coboundary_attr=v.kwargs['coboundary_attr'])

    expected_up_msg = torch.tensor([[6], [4], [11], [9], [7]], dtype=torch.float)
    assert torch.equal(up_msg, expected_up_msg)

    expected_down_msg = torch.zeros(5, 1)
    assert torch.equal(down_msg, expected_down_msg)

    expected_boundary_msg = torch.zeros(5, 1)
    assert torch.equal(boundary_msg, expected_boundary_msg)

    if include_coboundary_links:
        expected_coboundary_msg = torch.tensor([[5], [3], [11], [12], [11]], dtype=torch.float)
    else:
        expected_coboundary_msg = torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.float)
    assert torch.equal(coboundary_msg, expected_coboundary_msg)


@pytest.mark.parametrize("include_coboundary_links", [True, False])
def test_propagate_at_two_cell_level_in_cmp_when_there_is_a_single_one(include_coboundary_links):
    """We build a graph in the shape of a house (a triangle on top of a square)
    and test propagation at the two_cell level. This makes sure that propagate works when
    up_index is None."""

    house_complex = get_house_complex(include_coboundary_links)
    t = house_complex.get_cochain_params(dim=2)

    # Extract the message passing object and propagate
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1,
                 use_coboundary_msg=include_coboundary_links)
    up_msg, down_msg, boundary_msg, coboundary_msg = cmp.propagate(t.up_index, t.down_index,
                                               t.boundary_index, None, x=t.x,
                                               up_attr=t.kwargs['up_attr'],
                                               down_attr=t.kwargs['down_attr'],
                                               boundary_attr=t.kwargs['boundary_attr'],
                                               coboundary_attr=t.kwargs['coboundary_attr'])

    expected_up_msg = torch.zeros(1, 1)
    assert torch.equal(up_msg, expected_up_msg)

    expected_down_msg = torch.zeros(1, 1)
    assert torch.equal(down_msg, expected_down_msg)

    expected_boundary_msg = torch.tensor([[14]], dtype=torch.float)
    assert torch.equal(boundary_msg, expected_boundary_msg)
    
    if include_coboundary_links:
        expected_coboundary_msg = torch.tensor([[0]], dtype=torch.float)
    else:
        expected_coboundary_msg = torch.tensor([[0]], dtype=torch.float)
    assert torch.equal(coboundary_msg, expected_coboundary_msg)


@pytest.mark.parametrize("include_coboundary_links", [True, False])
def test_edge_propagate_at_edge_level_in_cmp_2(include_coboundary_links):
    """We test propagation at the edge level."""

    molecular_complex = get_molecular_complex(include_coboundary_links)
    e = molecular_complex.get_cochain_params(dim=1)
    assert e.kwargs['boundary_index'] is not None, e.kwargs['boundary_index']

    # Extract the message passing object and propagate
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1,
                 use_coboundary_msg=include_coboundary_links)
    up_msg, down_msg, boundary_msg, coboundary_msg = cmp.propagate(e.up_index, e.down_index,
                                               e.boundary_index, e.coboundary_index, x=e.x,
                                               up_attr=e.kwargs['up_attr'],
                                               down_attr=e.kwargs['down_attr'],
                                               boundary_attr=e.kwargs['boundary_attr'],
                                               coboundary_attr=e.kwargs['coboundary_attr'])
    expected_down_msg = torch.tensor([[14], [17], [11], [4], [11], [12], [23], [19], [15]], dtype=torch.float)
    assert torch.equal(down_msg, expected_down_msg)

    expected_up_msg = torch.tensor([[9], [34], [7], [6], [23], [22], [21], [20], [0]], dtype=torch.float)
    assert torch.equal(up_msg, expected_up_msg)

    expected_boundary_msg = torch.tensor([[3], [5], [7], [5], [8], [11], [13], [9], [15]], dtype=torch.float)
    assert torch.equal(boundary_msg, expected_boundary_msg)

    if include_coboundary_links:
        expected_coboundary_msg = torch.tensor([[1], [3], [1], [1], [2], [2], [2], [2], [0]], dtype=torch.float)
    else:
        expected_coboundary_msg = torch.tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype=torch.float)
    assert torch.equal(coboundary_msg, expected_coboundary_msg)
        

@pytest.mark.parametrize("include_coboundary_links", [True, False])
def test_propagate_at_vertex_level_in_cmp_2(include_coboundary_links):
    """We test propagation at the vertex level. This makes sure propagate works when
    down_index is None.
    """

    molecular_complex = get_molecular_complex(include_coboundary_links)
    v = molecular_complex.get_cochain_params(dim=0)
    ###################### FIX: v.coboundary_attr is None here... It shouldn't be
    # Extract the message passing object and propagate
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1,
                 use_coboundary_msg=include_coboundary_links)
    up_msg, down_msg, boundary_msg, coboundary_msg = cmp.propagate(v.up_index, v.down_index,
                                               v.boundary_index, v.coboundary_index, x=v.x,
                                               up_attr=v.kwargs['up_attr'],
                                               down_attr=v.kwargs['down_attr'],
                                               boundary_attr=v.kwargs['boundary_attr'],
                                               coboundary_attr=v.kwargs['coboundary_attr'])

    expected_up_msg = torch.tensor([[6], [11], [11], [4], [9], [12], [16], [7]], dtype=torch.float)
    assert torch.equal(up_msg, expected_up_msg)

    expected_down_msg = torch.zeros(8, 1)
    assert torch.equal(down_msg, expected_down_msg)

    expected_boundary_msg = torch.zeros(8, 1)
    assert torch.equal(boundary_msg, expected_boundary_msg)

    if include_coboundary_links:
        expected_coboundary_msg = torch.tensor([[5], [11], [10], [7], [11], [13], [24], [9]], dtype=torch.float)
    else:
        expected_coboundary_msg = torch.tensor([[0], [0], [0], [0], [0], [0], [0], [0]], dtype=torch.float)
    assert torch.equal(coboundary_msg, expected_coboundary_msg)


@pytest.mark.parametrize("include_coboundary_links", [True, False])
def test_propagate_at_two_cell_level_in_cmp(include_coboundary_links):
    """We test propagation at the two_cell level"""

    molecular_complex = get_molecular_complex(include_coboundary_links)
    t = molecular_complex.get_cochain_params(dim=2)

    # Extract the message passing object and propagate
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1,
                 use_coboundary_msg=include_coboundary_links)
    up_msg, down_msg, boundary_msg, coboundary_msg = cmp.propagate(t.up_index, t.down_index,
                                               t.boundary_index, None, x=t.x,
                                               up_attr=t.kwargs['up_attr'],
                                               down_attr=t.kwargs['down_attr'],
                                               boundary_attr=t.kwargs['boundary_attr'],
                                               coboundary_attr=t.kwargs['coboundary_attr'])

    expected_up_msg = torch.zeros(2, 1)
    assert torch.equal(up_msg, expected_up_msg)

    expected_down_msg = torch.tensor([[2], [2]], dtype=torch.float)
    assert torch.equal(down_msg, expected_down_msg)

    expected_boundary_msg = torch.tensor([[10], [18]], dtype=torch.float)
    assert torch.equal(boundary_msg, expected_boundary_msg)
    
    if include_coboundary_links:
        expected_coboundary_msg = torch.zeros(2, 1)
    else:
        expected_coboundary_msg = torch.zeros(2, 1)
    assert torch.equal(coboundary_msg, expected_coboundary_msg)


def test_propagate_at_two_cell_level_in_cmp():
    """We build a graph formed of two triangles sharing an edge.
    This makes sure that propagate works when up_index is None."""
    # TODO: Refactor this test to use the kite complex

    # When there is a single two_cell, there is no upper or lower adjacency
    up_index = None
    down_index = torch.tensor([[0, 1],
                               [1, 0]], dtype=torch.long)
    # Add features for the edges shared by the triangles
    down_attr = torch.tensor([[1], [1]])

    # We initialise the vertices with dummy scalar features
    x = torch.tensor([[32], [17]], dtype=torch.float)

    # Extract the message passing object and propagate
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, down_msg, _, _ = cmp.propagate(up_index, down_index, None, None, x=x, down_attr=down_attr)
    expected_updated_x = torch.tensor([[17], [32]], dtype=torch.float)

    assert torch.equal(up_msg + down_msg, expected_updated_x)


def test_smp_messaging_with_isolated_nodes():
    """
    This checks how pyG handles messages for isolated nodes. This shows that it sends a zero vector.
    """
    square_dot_complex = get_square_dot_complex()
    params = square_dot_complex.get_cochain_params(dim=0)

    mp = MessagePassing()
    out = mp.propagate(edge_index=params.up_index, x=params.x)
    isolated_out = out[4]

    # This confirms pyG returns a zero message to isolated vertices
    assert torch.equal(isolated_out, torch.zeros_like(isolated_out))
    for i in range(4):
        assert not torch.equal(out[i], torch.zeros_like(out[i]))

    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, down_msg, _, _ = cmp.propagate(up_index=params.up_index, down_index=None, boundary_index=None,
                                        coboundary_index=None,
                                        x=params.x, up_attr=None)
    assert torch.equal(out, up_msg)
    assert torch.equal(down_msg, torch.zeros_like(down_msg))


def test_cmp_messaging_with_isolated_node_only():
    """
    This checks how pyG handles messages for one isolated node.
    """
    fullstop_complex = get_fullstop_complex()
    params = fullstop_complex.get_cochain_params(dim=0)
    empty_edge_index = torch.LongTensor([[],[]])

    mp = MessagePassing()
    mp_out = mp.propagate(edge_index=empty_edge_index, x=params.x)

    # This confirms pyG returns a zero message when edge_index is empty
    assert torch.equal(mp_out, torch.zeros_like(mp_out))

    # This confirms behavior is consistent with our framework
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, _, _, _ = cmp.propagate(up_index=params.up_index, down_index=None, boundary_index=None,
                                        coboundary_index=None,
                                        x=params.x, up_attr=None)
    assert torch.equal(mp_out, up_msg)


def test_cmp_messaging_with_two_isolated_nodes_only():
    """
    This checks how pyG handles messages for two isolated nodes.
    """
    colon_complex = get_colon_complex()
    params = colon_complex.get_cochain_params(dim=0)
    empty_edge_index = torch.LongTensor([[],[]])

    mp = MessagePassing()
    mp_out = mp.propagate(edge_index=empty_edge_index, x=params.x)
    
    # This confirms pyG returns a zero message when edge_index is empty
    assert torch.equal(mp_out, torch.zeros_like(mp_out))

    # This confirms behavior is consistent with our framework
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1)
    up_msg, _, _, _ = cmp.propagate(up_index=params.up_index, down_index=None, boundary_index=None,
                                        coboundary_index=None,
                                        x=params.x, up_attr=None)
    assert torch.equal(mp_out, up_msg)


def test_cmp_messaging_with_replicated_adjs():
    """
    This checks message passing works as expected in case cells/simplices
    share more than one (co)boundary.
    """
    bridged_complex = get_bridged_complex()
    bridged_graph = convert_to_graph(bridged_complex)
    bridged_complex_from_graph = compute_ring_2complex(
        bridged_graph.x, bridged_graph.edge_index, bridged_graph.edge_attr, bridged_graph.num_nodes,
        bridged_graph.y, init_method='sum', init_edges=True, init_rings=True)
    check_edge_index_are_the_same(bridged_complex_from_graph.edges.upper_index, bridged_complex.edges.upper_index)
    check_edge_index_are_the_same(bridged_complex_from_graph.two_cells.lower_index, bridged_complex.two_cells.lower_index)
    check_edge_attr_are_the_same(bridged_complex.cochains[1].boundary_index, bridged_complex.cochains[1].x, bridged_graph.edge_index, bridged_graph.edge_attr)
    check_edge_attr_are_the_same(bridged_complex_from_graph.cochains[1].boundary_index, bridged_complex_from_graph.cochains[1].x, bridged_graph.edge_index, bridged_graph.edge_attr)
    
    # verify up-messaging with multiple shared coboundaries
    e = bridged_complex.get_cochain_params(dim=1)
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1)
    e_up_msg, e_down_msg, e_boundary_msg, _ = cmp.propagate(e.up_index, e.down_index,
                                               e.boundary_index, None, x=e.x,
                                               up_attr=e.kwargs['up_attr'],
                                               down_attr=e.kwargs['down_attr'],
                                               boundary_attr=e.kwargs['boundary_attr'])
    expected_e_up_msg = torch.tensor([[4+5+6+2+3+4],  # edge 0
                                      [3+5+6+1+3+4],  # edge 1
                                      [2+5+6+1+2+4],  # edge 2
                                      [1+5+6+1+2+3],  # edge 3
                                      [1+4+6+2+3+6],  # edge 4
                                      [1+4+5+2+3+5]], # edge 5
                                      dtype=torch.float)
    assert torch.equal(e_up_msg, expected_e_up_msg)
    
    # same but start from graph instead
    e = bridged_complex_from_graph.get_cochain_params(dim=1)
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1)
    e_up_msg, e_down_msg, e_boundary_msg, _ = cmp.propagate(e.up_index, e.down_index,
                                               e.boundary_index, None, x=e.x,
                                               up_attr=e.kwargs['up_attr'],
                                               down_attr=e.kwargs['down_attr'],
                                               boundary_attr=e.kwargs['boundary_attr'])
    
    expected_e_up_msg = torch.tensor([[4+5+6+2+3+4],        # edge 0-1 (0)
                                      [1+5+6+1+2+3],        # edge 0-3 (3)
                                      [3+5+6+1+3+4],        # edge 1-2 (1)
                                      [1+4+5+2+3+5],        # edge 1-4 (5)
                                      [2+5+6+1+2+4],        # edge 2-3 (2)
                                      [1+4+6+2+3+6]],       # edge 3-4 (4)
                                      dtype=torch.float)
    assert torch.equal(e_up_msg, expected_e_up_msg)
    
    # verify down-messaging with multiple shared boundaries
    t = bridged_complex.get_cochain_params(dim=2)
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1)
    t_up_msg, t_down_msg, t_boundary_msg, _ = cmp.propagate(t.up_index, t.down_index,
                                               t.boundary_index, None, x=t.x,
                                               up_attr=t.kwargs['up_attr'],
                                               down_attr=t.kwargs['down_attr'],
                                               boundary_attr=t.kwargs['boundary_attr'])
    expected_t_down_msg = torch.tensor([[2+2+3+3],    # ring 0
                                        [1+1+3+3],    # ring 1
                                        [1+1+2+2]],   # ring 2
                                      dtype=torch.float)
    assert torch.equal(t_down_msg, expected_t_down_msg)
    
    expected_t_boundary_msg = torch.tensor([[1+6+5+4],    # ring 0
                                        [2+3+5+6],    # ring 1
                                        [1+2+3+4]],   # ring 2
                                      dtype=torch.float)
    assert torch.equal(t_boundary_msg, expected_t_boundary_msg)
    
    # same but start from graph instead
    t = bridged_complex_from_graph.get_cochain_params(dim=2)
    cmp = CochainMessagePassing(up_msg_size=1, down_msg_size=1)
    t_up_msg, t_down_msg, t_boundary_msg, _ = cmp.propagate(t.up_index, t.down_index,
                                               t.boundary_index, None, x=t.x,
                                               up_attr=t.kwargs['up_attr'],
                                               down_attr=t.kwargs['down_attr'],
                                               boundary_attr=t.kwargs['boundary_attr'])
    t0_x = 1+2+4+5  # 12
    t1_x = 2+3+4+5  # 14
    t2_x = 1+2+3+4  # 10
    expected_t_down_msg = torch.tensor([[t1_x + t1_x + t2_x + t2_x],   # ring 0-1-4-3 (0)
                                        [t0_x + t0_x + t1_x + t1_x],   # ring 0-1-2-3 (2)
                                        [t0_x + t0_x + t2_x + t2_x]],  # ring 1-2-3-4 (1)
                                      dtype=torch.float)
    assert torch.equal(t_down_msg, expected_t_down_msg)
    
    expected_t_boundary_msg = torch.tensor([[1+6+5+4],        # ring 0
                                        [1+2+3+4],        # ring 2
                                        [2+3+5+6]],       # ring 1
                                      dtype=torch.float)
    assert torch.equal(t_boundary_msg, expected_t_boundary_msg)


# TODO: test compute_ring_2complex or something similar in data.utils with coboundary computation True