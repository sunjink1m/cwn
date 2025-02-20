import torch
import itertools
import pytest

from data.complex import ComplexBatch
from data.dummy_complexes import get_testing_complex_list
from mp.molec_models import (OGBEmbedCWN, EmbedSparseCINNoRings, 
                            EmbedGIN, EmbedDenseCIN)
from mp.layers import DenseBasicConv
from data.data_loading import DataLoader, load_dataset


def test_embed_sparse_cin_no_rings_model_with_batching():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1]
    bs = list(range(2, len(data_list)+1))
    params = itertools.product(bs, dims, dims)
    torch.manual_seed(0)
    for batch_size, batch_max_dim, model_max_dim in params:
        if batch_max_dim > model_max_dim:
            continue

        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)
        model = EmbedSparseCINNoRings(atom_types=32, bond_types=4, out_size=3, num_layers=3, hidden=5)
        # We use the model in eval mode to avoid problems with batch norm.
        model.eval()

        batched_res = []
        for batch in data_loader:
            # Simulate no edge and two_cell features to test init layer
            if len(batch.cochains) >= 2:
                batch.cochains[1].x = None
            if len(batch.cochains) == 3:
                batch.cochains[2].x = None

            batched_pred = model.forward(batch)
            batched_res.append(batched_pred)

        batched_res = torch.cat(batched_res, dim=0)

        unbatched_res = []
        for complex in data_list:
            batch = ComplexBatch.from_complex_list([complex], max_dim=batch_max_dim)

            # Simulate no edge and two_cell features to test init layer
            if len(batch.cochains) >= 2:
                batch.cochains[1].x = None
            if len(batch.cochains) == 3:
                batch.cochains[2].x = None

            pred = model.forward(batch)
            unbatched_res.append(pred)

        unbatched_res = torch.cat(unbatched_res, dim=0)
        assert torch.allclose(unbatched_res, batched_res, atol=1e-6)


def test_embed_gin_model_with_batching():
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1]
    bs = list(range(2, len(data_list)+1))
    params = itertools.product(bs, dims, dims)
    torch.manual_seed(0)
    for batch_size, batch_max_dim, model_max_dim in params:
        if batch_max_dim > model_max_dim:
            continue

        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)
        model = EmbedGIN(atom_types=32, bond_types=4, out_size=3, num_layers=3, hidden=5)
        # We use the model in eval mode to avoid problems with batch norm.
        model.eval()

        batched_res = []
        for batch in data_loader:
            # Simulate no edge and two_cell features to test init layer
            if len(batch.cochains) >= 2:
                batch.cochains[1].x = None
            if len(batch.cochains) == 3:
                batch.cochains[2].x = None

            batched_pred = model.forward(batch)
            batched_res.append(batched_pred)

        batched_res = torch.cat(batched_res, dim=0)

        unbatched_res = []
        for complex in data_list:
            batch = ComplexBatch.from_complex_list([complex], max_dim=batch_max_dim)

            # Simulate no edge and two_cell features to test init layer
            if len(batch.cochains) >= 2:
                batch.cochains[1].x = None
            if len(batch.cochains) == 3:
                batch.cochains[2].x = None

            pred = model.forward(batch)
            unbatched_res.append(pred)

        unbatched_res = torch.cat(unbatched_res, dim=0)
        assert torch.allclose(unbatched_res, batched_res, atol=1e-6)


@pytest.mark.parametrize("model", ['sparsecin','lesssparsecin','densecin'])
@pytest.mark.data
def test_molec_models_with_batching_on_proteins(model):
    if model == 'sparsecin':
        include_down_adj = False
        include_coboundary_links = False
        model = EmbedDenseCIN(atom_types=64, bond_types=4, out_size=3, num_layers=3, hidden=5,
                           jump_mode='cat', max_dim=2,
                            use_up_attr=True, variant='dense')
    elif model == 'lesssparsecin':
        include_down_adj = True
        include_coboundary_links = False
        model = EmbedDenseCIN(atom_types=64, bond_types=4, out_size=3, num_layers=3, hidden=5,
                           jump_mode='cat', max_dim=2,
                            use_up_attr=True,
                            use_down_attr=True, variant='less-sparse')
    elif model == 'densecin':
        include_down_adj = True
        include_coboundary_links = True
        model = EmbedDenseCIN(atom_types=64, bond_types=4, out_size=3,
                            num_layers=3, hidden=5,
                            dropout_rate=0.5, 
                            max_dim=2, 
                            jump_mode=None, 
                            nonlinearity='sigmoid',
                            embed_edge=True, 
                            embed_dim=5, 
                            use_up_attr=True,
                            use_down_attr=True)

    model.eval()

    """Check this runs without errors and that batching and no batching produce the same output."""
    dataset = load_dataset('PROTEINS', max_dim=2, fold=0, init_method='mean',
                        include_down_adj=include_down_adj,
                        include_coboundary_links=include_coboundary_links)
    assert len(dataset) == 1113
    split_idx = dataset.get_idx_split()
    dataset = dataset[split_idx['valid']]
    assert len(dataset) == 111

    max_dim = 2
    torch.manual_seed(0)
    data_loader = DataLoader(dataset, batch_size=32, max_dim=max_dim)
    

    batched_res = {}
    for batch in data_loader:
        # Simulate no edge and two_cell features to test init layer
        batch.cochains[1].x = None
        if len(batch.cochains) == 3:
            batch.cochains[2].x = None
        # ZincSparseCIN assumes features are unidimensional like in ZINC
        batch.cochains[0].x = batch.cochains[0].x[:, :1]

        batched_pred, res = model.forward(batch, include_partial=True)
        for key in res:
            if key not in batched_res:
                batched_res[key] = []
            batched_res[key].append(res[key])

    for key in batched_res:
        batched_res[key] = torch.cat(batched_res[key], dim=0)

    unbatched_res = {}
    for complex in dataset:
        batch = ComplexBatch.from_complex_list([complex], max_dim=max_dim)
        # Simulate no edge and two_cell features to test init layer
        batch.cochains[1].x = None
        if len(batch.cochains) == 3:
            batch.cochains[2].x = None
        # ZincSparseCIN assumes features are unidimensional like in ZINC
        batch.cochains[0].x = batch.cochains[0].x[:, :1]

        pred, res = model.forward(batch, include_partial=True)
        for key in res:
            if key not in unbatched_res:
                unbatched_res[key] = []
            unbatched_res[key].append(res[key])

    for key in unbatched_res:
        unbatched_res[key] = torch.cat(unbatched_res[key], dim=0)

    for key in set(list(unbatched_res.keys()) + list(batched_res.keys())):
        assert torch.allclose(unbatched_res[key], batched_res[key], atol=1e-6), (
                print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))


@pytest.mark.parametrize("model", ['ogbsparse', 'ogbbasic','sparse', 'lesssparse','zincdensecin'])
def test_molec_models_with_batching(model):
    """Check this runs without errors and that batching and no batching produce the same output."""
    data_list = get_testing_complex_list()

    # Try multiple parameters
    dims = [1, 2]
    bs = list(range(2, len(data_list)+1))
    params = itertools.product(bs, dims, dims)
    torch.manual_seed(0)
    for batch_size, batch_max_dim, model_max_dim in params:
        if batch_max_dim > model_max_dim:
            continue

        data_loader = DataLoader(data_list, batch_size=batch_size, max_dim=batch_max_dim)
        if model=='ogbsparse':
            model = OGBEmbedCWN(out_size=3, num_layers=3, hidden=5,
                                    jump_mode=None, max_dim=model_max_dim)
        elif model=='ogbbasic':
            model = OGBEmbedCWN(out_size=3, num_layers=3, hidden=5,
                                    jump_mode=None, max_dim=model_max_dim, conv_type=DenseBasicConv)
        elif model=='sparse':
            model = EmbedDenseCIN(atom_types=32, bond_types=4, out_size=3, num_layers=3, hidden=5,
                                jump_mode='cat', max_dim=model_max_dim, variant='sparse')
        elif model=='lesssparse':
            model = EmbedDenseCIN(atom_types=32, bond_types=4, out_size=3, num_layers=3, hidden=5,
                                jump_mode='cat', max_dim=model_max_dim, variant='less-sparse')
        elif model=='zincdensecin':
            model = EmbedDenseCIN(atom_types=32, 
                        bond_types=4, 
                        out_size=3, 
                        num_layers=3, hidden=5,
                        dropout_rate=0.5, 
                        max_dim=model_max_dim, 
                        jump_mode=None, 
                        nonlinearity='sigmoid',
                        embed_edge=True, 
                        embed_dim=5,
                        variant='dense')
        # We use the model in eval mode to avoid problems with batch norm.
        model.eval()

        batched_res = {}
        for batch in data_loader:
            # Simulate no edge and two_cell features to test init layer
            if len(batch.cochains) >= 2:
                batch.cochains[1].x = None
            if len(batch.cochains) == 3:
                batch.cochains[2].x = None

            batched_pred, res = model.forward(batch, include_partial=True)
            for key in res:
                if key not in batched_res:
                    batched_res[key] = []
                batched_res[key].append(res[key])

        for key in batched_res:
            batched_res[key] = torch.cat(batched_res[key], dim=0)

        unbatched_res = {}
        for complex in data_list:
            batch = ComplexBatch.from_complex_list([complex], max_dim=batch_max_dim)

            # Simulate no edge and two_cell features to test init layer
            if len(batch.cochains) >= 2:
                batch.cochains[1].x = None
            if len(batch.cochains) == 3:
                batch.cochains[2].x = None

            pred, res = model.forward(batch, include_partial=True)

            for key in res:
                if key not in unbatched_res:
                    unbatched_res[key] = []
                unbatched_res[key].append(res[key])

        for key in unbatched_res:
            unbatched_res[key] = torch.cat(unbatched_res[key], dim=0)

        for key in set(list(unbatched_res.keys()) + list(batched_res.keys())):
            assert torch.allclose(unbatched_res[key], batched_res[key], atol=1e-6), (
                    print(key, torch.max(torch.abs(unbatched_res[key] - batched_res[key]))))
