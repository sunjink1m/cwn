import torch
from torch import nn
import torch.optim as optim

from mp.nn import get_nonlinearity, get_graph_norm
from mp.layers import (
    DummyCellularMessagePassing, CINConv, OrientedConv, InitReduceConv, EmbedVEWithReduce,  DenseCINConv, 
    SparseDeeperCCNConv)
from data.dummy_complexes import get_house_complex, get_molecular_complex, get_bridged_complex, get_filled_square_complex
from torch import nn
from data.datasets.flow import load_flow_dataset
from data.complex import ComplexBatch


def test_dummy_cellular_message_passing_with_down_msg():
    house_complex = get_house_complex()
    v_params = house_complex.get_cochain_params(dim=0)
    e_params = house_complex.get_cochain_params(dim=1)
    t_params = house_complex.get_cochain_params(dim=2)

    dsmp = DummyCellularMessagePassing()
    v_x, e_x, t_x = dsmp.forward(v_params, e_params, t_params)

    expected_v_x = torch.tensor([[12], [9], [25], [25], [23]], dtype=torch.float)
    assert torch.equal(v_x, expected_v_x)

    expected_e_x = torch.tensor([[10], [20], [47], [22], [42], [37]], dtype=torch.float)
    assert torch.equal(e_x, expected_e_x)

    expected_t_x = torch.tensor([[1]], dtype=torch.float)
    assert torch.equal(t_x, expected_t_x)


def test_dummy_cellular_message_passing_with_boundary_msg():
    house_complex = get_house_complex()
    v_params = house_complex.get_cochain_params(dim=0)
    e_params = house_complex.get_cochain_params(dim=1)
    t_params = house_complex.get_cochain_params(dim=2)

    dsmp = DummyCellularMessagePassing(use_boundary_msg=True, use_down_msg=False)
    v_x, e_x, t_x = dsmp.forward(v_params, e_params, t_params)

    expected_v_x = torch.tensor([[12], [9], [25], [25], [23]], dtype=torch.float)
    assert torch.equal(v_x, expected_v_x)

    expected_e_x = torch.tensor([[4], [7], [23], [9], [25], [24]], dtype=torch.float)
    assert torch.equal(e_x, expected_e_x)

    expected_t_x = torch.tensor([[15]], dtype=torch.float)
    assert torch.equal(t_x, expected_t_x)


def test_dummy_cellular_message_passing_on_molecular_cell_complex():
    molecular_complex = get_molecular_complex()
    v_params = molecular_complex.get_cochain_params(dim=0)
    e_params = molecular_complex.get_cochain_params(dim=1)
    ring_params = molecular_complex.get_cochain_params(dim=2)

    dsmp = DummyCellularMessagePassing(use_boundary_msg=True, use_down_msg=True)
    v_x, e_x, ring_x = dsmp.forward(v_params, e_params, ring_params)

    expected_v_x = torch.tensor([[12], [24], [24], [15], [25], [31], [47], [24]],
        dtype=torch.float)
    assert torch.equal(v_x, expected_v_x)

    expected_e_x = torch.tensor([[35], [79], [41], [27], [66], [70], [92], [82], [53]],
        dtype=torch.float)
    assert torch.equal(e_x, expected_e_x)

    # The first cell feature is given by 1[x] + 0[up] + (2+2)[down] + (1+2+3+4)[boundaries] = 15
    # The 2nd cell is given by 2[x] + 0[up] + (1+2)[down] + (2+5+6+7+8)[boundaries] = 33
    expected_ring_x = torch.tensor([[15], [33]], dtype=torch.float)
    assert torch.equal(ring_x, expected_ring_x)


def test_cin_conv_training():
    msg_net = nn.Sequential(nn.Linear(2, 1))
    update_net = nn.Sequential(nn.Linear(1, 3))

    cin_conv = CINConv(1, 1, msg_net, msg_net, update_net, 0.05)

    all_params_before = []
    for p in cin_conv.parameters():
        all_params_before.append(p.clone().data)
    assert len(all_params_before) > 0

    house_complex = get_house_complex()

    v_params = house_complex.get_cochain_params(dim=0)
    e_params = house_complex.get_cochain_params(dim=1)
    t_params = house_complex.get_cochain_params(dim=2)

    yv = house_complex.get_labels(dim=0)
    ye = house_complex.get_labels(dim=1)
    yt = house_complex.get_labels(dim=2)
    y = torch.cat([yv, ye, yt])

    optimizer = optim.SGD(cin_conv.parameters(), lr=0.001)
    optimizer.zero_grad()

    out_v, out_e, out_t = cin_conv.forward(v_params, e_params, t_params)
    out = torch.cat([out_v, out_e, out_t], dim=0)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    all_params_after = []
    for p in cin_conv.parameters():
        all_params_after.append(p.clone().data)
    assert len(all_params_after) == len(all_params_before)

    # Check that parameters have been updated.
    for i, _ in enumerate(all_params_before):
        assert not torch.equal(all_params_before[i], all_params_after[i])


def test_orient_conv_on_flow_dataset():
    import numpy as np

    np.random.seed(4)
    update_up = nn.Sequential(nn.Linear(1, 4))
    update_down = nn.Sequential(nn.Linear(1, 4))
    update = nn.Sequential(nn.Linear(1, 4))

    train, _, G = load_flow_dataset(num_points=400, num_train=3, num_test=3)
    number_of_edges = G.number_of_edges()

    model = OrientedConv(1, 1, 1, update_up_nn=update_up, update_down_nn=update_down,
        update_nn=update, act_fn=torch.tanh)
    model.eval()

    out = model.forward(train[0])
    assert out.size(0) == number_of_edges
    assert out.size(1) == 4


def test_init_reduce_conv_on_house_complex():
    house_complex = get_house_complex()
    v_params = house_complex.get_cochain_params(dim=0)
    e_params = house_complex.get_cochain_params(dim=1)
    t_params = house_complex.get_cochain_params(dim=2)

    conv = InitReduceConv(reduce='add')

    ex = conv.forward(v_params.x, e_params.boundary_index)
    expected_ex = torch.tensor([[3], [5], [7], [5], [9], [8]], dtype=torch.float)
    assert torch.equal(expected_ex, ex)

    tx = conv.forward(e_params.x, t_params.boundary_index)
    expected_tx = torch.tensor([[14]], dtype=torch.float)
    assert torch.equal(expected_tx, tx)


def test_embed_with_reduce_layer_on_house_complex():
    house_complex = get_house_complex()
    cochains = house_complex.cochains
    params = house_complex.get_all_cochain_params()

    embed_layer = nn.Embedding(num_embeddings=32, embedding_dim=10)
    init_reduce = InitReduceConv()
    conv = EmbedVEWithReduce(embed_layer, None, init_reduce)

    # Simulate the lack of features in these dimensions.
    params[1].x = None
    params[2].x = None

    xs = conv.forward(*params)

    assert len(xs) == 3
    assert xs[0].dim() == 2
    assert xs[0].size(0) == cochains[0].num_cells
    assert xs[0].size(1) == 10
    assert xs[1].size(0) == cochains[1].num_cells
    assert xs[1].size(1) == 10
    assert xs[2].size(0) == cochains[2].num_cells
    assert xs[2].size(1) == 10



def test_dense_cin_conv_training():
    '''
    This testmakes sure the layers that should be used gets used and ones that 
    shouldn't aren't being used.
    
    (For example, down-adjacency-message-passing neural network is not used in 
    the 0th dimension because there are no down-adjacencies in the 0th dimension)

    It does this by doing one backpropagation step and seeing which parameters 
    have changed.

    Therefore this test makes sure cellular message passing is working properly.
    '''
    house_complex = get_house_complex(include_coboundary_links=True)
    molecular_complex = get_molecular_complex(include_coboundary_links=True)
    bridged_complex = get_bridged_complex()
    filled_square = get_filled_square_complex()

    batch = ComplexBatch.from_complex_list([house_complex, molecular_complex, bridged_complex, filled_square])

    v_params = batch.get_cochain_params(dim=0, include_coboundary_features=True)
    e_params = batch.get_cochain_params(dim=1, include_coboundary_features=True)
    t_params = batch.get_cochain_params(dim=2, include_coboundary_features=True)

    yv = batch.get_labels(dim=0)
    ye = batch.get_labels(dim=1)
    yt = batch.get_labels(dim=2)
    y = torch.cat([yv, ye, yt])


    conv = DenseCINConv(up_msg_size=1, 
                        down_msg_size=1,
                        boundary_msg_size=1,
                        coboundary_msg_size=1, 
                        passed_msg_up_nn=None,
                        passed_msg_down_nn=None,
                        passed_msg_boundaries_nn=None,
                        passed_msg_coboundaries_nn=None,
                        passed_update_up_nn=None,
                        passed_update_down_nn=None,
                        passed_update_coboundaries_nn=None,
                        passed_update_boundaries_nn=None,
                        train_eps=True,
                        hidden=3, 
                        act_module=get_nonlinearity('id', return_module=True), 
                        layer_dim=1,
                        graph_norm=get_graph_norm('id'), 
                        use_coboundaries=True,
                        use_boundaries=True)


    # these linear layers should be used during forward calculation
    active_layers = [
                    conv.mp_levels[0].msg_up_nn,
                    conv.mp_levels[0].msg_coboundaries_nn,
                    conv.mp_levels[0].update_up_nn,
                    conv.mp_levels[0].update_coboundaries_nn,

                    conv.mp_levels[1].msg_up_nn,
                    conv.mp_levels[1].msg_coboundaries_nn,
                    conv.mp_levels[1].update_up_nn,
                    conv.mp_levels[1].update_coboundaries_nn,
                    conv.mp_levels[1].msg_down_nn,
                    conv.mp_levels[1].msg_boundaries_nn,
                    conv.mp_levels[1].update_down_nn,
                    conv.mp_levels[1].update_boundaries_nn,
                    
                    conv.mp_levels[2].msg_down_nn,
                    conv.mp_levels[2].msg_boundaries_nn,
                    conv.mp_levels[2].update_down_nn,
                    conv.mp_levels[2].update_boundaries_nn,
                    ]
    
    # these linear layers should ignored during forward calculation
    inactive_layers = [
                    conv.mp_levels[0].msg_down_nn,
                    conv.mp_levels[0].msg_boundaries_nn,
                    # conv.mp_levels[0].update_down_nn, # <- so apparently all the update nn's get used anyways, 
                                                        # even if the corresponding adjacencies don't exist
                                                        # because we add (1+eps)*cochain.x to a zero vector 
                                                        # that came out of .propagate and send it into the 
                                                        # update nn
                    # conv.mp_levels[0].update_boundaries_nn,

                    conv.mp_levels[2].msg_up_nn,
                    conv.mp_levels[2].msg_coboundaries_nn,
                    # conv.mp_levels[2].update_up_nn,
                    # conv.mp_levels[2].update_coboundaries_nn,
                    ]

    # this list stores the values of the initial weights
    all_active_params_before = []
    # this list keeps track of which layers the weights come from
    for layer in active_layers:
        try:
            for parameter in layer.parameters():
                all_active_params_before.append(parameter.clone().data)
        except:
            print(f'{layer} is not a nn.Module')
    assert len(all_active_params_before) > 0

    all_inactive_params_before = []
    for layer in inactive_layers:
        try:
            for parameter in layer.parameters():
                all_inactive_params_before.append(parameter.clone().data)
        except:
            print(f'{layer} is not a nn.Module')
    assert len(all_inactive_params_before) == 0 # we have removed all redundant 
                                                # parameters from DenseCINConv



    optimizer = optim.SGD(conv.parameters(), lr=10.)
    optimizer.zero_grad()

    out_v, out_e, out_t = conv.forward(v_params, e_params, t_params)
    out = torch.cat([out_v, out_e, out_t], dim=0).squeeze(1)

    criterion = nn.CrossEntropyLoss()
    # loss = criterion(out, y)
    loss = criterion(out, y)

    loss.backward()
    optimizer.step()



    all_active_params_after = []
    for layer in active_layers:
        try:
            for parameter in layer.parameters():
                all_active_params_after.append(parameter.clone().data)
        except:
            print(f'{layer} is not a nn.Module')
    assert len(all_active_params_after) > 0

    all_inactive_params_after = []
    for layer in inactive_layers:
        try:
            for parameter in layer.parameters():
                all_inactive_params_after.append(parameter.clone().data)
        except:
            print(f'{layer} is not a nn.Module')
    assert len(all_inactive_params_after) == 0

    asdf = [all_active_params_before[i]==all_active_params_after[i] for i in range(len(all_active_params_before))]
    asdf2 = [all_inactive_params_before[i]==all_inactive_params_after[i] for i in range(len(all_inactive_params_before))]
    # Check that parameters have been updated.
    for i, _ in enumerate(all_active_params_before):
        assert not torch.equal(all_active_params_before[i], all_active_params_after[i])

    # Check inactive layers have not been updated
    for i, _ in enumerate(all_inactive_params_before):
        assert torch.equal(all_inactive_params_before[i], all_inactive_params_after[i])
