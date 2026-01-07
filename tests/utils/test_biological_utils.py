import torch
from instanseg.utils.biological_utils import merge_nuclei_per_cell

def T(nuc, cell):
    return torch.tensor([[nuc, cell]])

def test_single_match_kept():
    nuc = [
        [1,1],
        [1,1],
    ]
    cell = [
        [1,1],
        [1,1],
    ]
    out = merge_nuclei_per_cell(T(nuc, cell), iou_thresh=0.5)
    assert torch.all(out[0,0] == 1)
    assert torch.all(out[0,1] == 1)

def test_unmatched_nucleus_removed():
    nuc = [
        [1,1],
        [0,0],
    ]
    cell = [
        [0,0],
        [2,2],
    ]
    out = merge_nuclei_per_cell(T(nuc, cell), iou_thresh=0.5)
    assert torch.all(out[0,0] == 0)
    assert torch.all(out[0,1] == 0)

def test_low_iou_rejected():
    nuc = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
    ]
    cell = [
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
    ]
    out = merge_nuclei_per_cell(T(nuc, cell), iou_thresh=0.6)
    assert torch.all(out[0,0] == 0)
    assert torch.all(out[0,1] == 0)

def test_multiple_nuclei_merged():
    nuc = [
        [1,0],
        [0,2],
    ]
    cell = [
        [3,3],
        [3,3],
    ]
    out = merge_nuclei_per_cell(T(nuc, cell), iou_thresh=0.5)
    assert torch.all(out[0, 0][0, 0] == 3)
    assert torch.all(out[0, 1][0, 0] == 3)

def test_split_overlap():
    nuc = [
        [1,1],
        [1,1],
    ]
    cell = [
        [2,2],
        [3,3],
    ]
    out = merge_nuclei_per_cell(T(nuc, cell), iou_thresh=0.5)
    assert out[0,0][0,0] == 2
    assert out[0,0][1,0] == 3

def test_duplicate_unnucleated_cell():
    nuc = [
        [0,0],
        [0,0],
    ]
    cell = [
        [4,4],
        [4,4],
    ]
    out = merge_nuclei_per_cell(
        T(nuc, cell),
        allow_unnucleated_cells=False
    )
    assert torch.all(out[0,0] == 1)
    assert torch.all(out[0,1] == 1)

def test_remove_unnucleated_cell():
    nuc = [
        [0,0],
        [0,0],
    ]
    cell = [
        [5,5],
        [5,5],
    ]
    inp = T(nuc, cell)
    out = merge_nuclei_per_cell(
        inp,
        allow_unnucleated_cells=True
    )
    assert torch.all(out == 0)

def test_mixed_cells():
    nuc = [
        [1,0],
        [0,0],
    ]
    cell = [
        [2,2],
        [3,3],
    ]
    out = merge_nuclei_per_cell(
        T(nuc, cell),
        iou_thresh=0.5,
        allow_unnucleated_cells=False
    )
    assert out[0,0][0,0] == 2
    assert out[0,0][1,0] == 3
    assert out[0,1][1,0] == 3
