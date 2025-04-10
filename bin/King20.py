import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from typing import List, Tuple, Union, Any, Mapping
import seqpro as sp
import torch.nn as nn
import seqmodels as sm


def king20(
    dataset = "SYN", 
    download_mouse=False,
    dataset_dir="./",
):
    urls_list = [
        "https://elifesciences.org/download/aHR0cHM6Ly9jZG4uZWxpZmVzY2llbmNlcy5vcmcvYXJ0aWNsZXMvNDEyNzkvZWxpZmUtNDEyNzktc3VwcDEtdjIueGxzeA--/elife-41279-supp1-v2.xlsx?_hash=nX7V5q5UXEGDbCoqhF23ru1RNUI14CBnHk27Cxlpgr4%3D",
        "https://elifesciences.org/download/aHR0cHM6Ly9jZG4uZWxpZmVzY2llbmNlcy5vcmcvYXJ0aWNsZXMvNDEyNzkvZWxpZmUtNDEyNzktc3VwcDItdjIueGxzeA--/elife-41279-supp2-v2.xlsx?_hash=V%2FARKgB5fn%2FUe1zKqVdBFHuq7na8rU%2BuFcWIQQwnAPM%3D",
        "https://elifesciences.org/download/aHR0cHM6Ly9jZG4uZWxpZmVzY2llbmNlcy5vcmcvYXJ0aWNsZXMvNDEyNzkvZWxpZmUtNDEyNzktc3VwcDQtdjIuZmFzdGE-/elife-41279-supp4-v2.fasta?_hash=HEFd7o2rJQ5Be4CHpN3L9SnE0zbrhhJah%2Fgao0poWG0%3D",
        "http://hgdownload.cse.ucsc.edu/goldenpath/mm10/bigZips/mm10.fa.gz"
    ]
    paths = [
        os.path.join(dataset_dir, "design.xlsx"),
        os.path.join(dataset_dir, "models.xlsx"),
        os.path.join(dataset_dir, "GEN.fasta"),
        os.path.join(dataset_dir, "mm10.fa.gz")
    ]
    
    # if the directory doesn't exist, create it
    os.makedirs(dataset_dir, exist_ok=True)

    # Design excel
    if not os.path.exists(paths[0]):
        print(f"Downloading king20 design spreadsheet to {paths[0]}")
        os.system("wget -q -O {0} {1}".format(paths[0], urls_list[0]))
        print(f"Finished downloading king20 design spreadsheet to {paths[0]}")
    else:
        print(f"Design spreadsheet already exists at {paths[0]}")
    
    # Model excel
    if not os.path.exists(paths[1]):
        print(f"Downloading king20 model spreadsheet to {paths[1]}")
        os.system("wget -q -O {0} {1}".format(paths[1], urls_list[1]))
        print(f"Finished downloading king20 spreadsheet to {paths[1]}")
    else:
        print(f"Model spreadsheet already exists at {paths[1]}")
    
    # Load in the synthetic dataset
    if dataset == "SYN":
        seq_tbl = pd.read_excel(paths[0], sheet_name=1)
        seq_tbl_filt = seq_tbl[~seq_tbl["Sequence"].duplicated()]
        model_tbl = pd.read_excel(paths[1], sheet_name=4)
        merged_tbl = pd.merge(seq_tbl_filt, model_tbl, on="Element_id", how="outer")
    
    # Load in the genomic dataset
    elif dataset == "GEN":
        seq_tbl = pd.read_excel(paths[0], sheet_name=6)
        seq_tbl_filt = seq_tbl[~seq_tbl["Element_id"].duplicated()]
        seq_tbl_wt = seq_tbl_filt[seq_tbl_filt["Element_id"].str.contains("Genomic")]
        exp_summary = pd.read_excel(paths[0], sheet_name=7)
        merged = pd.merge(seq_tbl_wt, exp_summary, on="Element_id", how="left")
        bc_grouped = merged.groupby("Element_id").agg("mean")
        CRE_norm_expression_WT_all = bc_grouped[["Rep1_Element_normalized", "Rep2_Element_normalized","Rep3_Element_normalized"]].mean(axis=1)
        bc_grouped["CRE_norm_expression_WT_all"] = CRE_norm_expression_WT_all
        bc_grouped["Sequence"] = merged.groupby("Element_id").agg({"Sequence": lambda x: x.iloc[0]})["Sequence"]
        merged_tbl = bc_grouped.reset_index()[["Sequence", "Element_id", "CRE_norm_expression_WT_all"]]
        merged_tbl["Element_id"] = merged_tbl["Element_id"].str.replace("_Genomic", "")
        merged_tbl["Barcode"] = "NA"
        
        if not os.path.exists(paths[2]):
            print(f"Downloading king20 gkmsvm fasta to {paths[2]}")
            os.system("wget -q -O {0} {1}".format(paths[2], urls_list[2]))
            print(f"Finished downloading king20 gkmsvm fasta to {paths[2]}")
        else:
            print(f"Gkmsvm fasta already exists at {paths[2]}")
        
        if download_mouse:
            if not os.path.exists(paths[3]):
                print(f"Downloading mm10 fasta to {paths[2]}")
                os.system("wget -q -O {0} {1}".format(paths[3], urls_list[3]))
                print(f"Finished downloading mm10 fasta to {paths[0]}")
            else:
                print(f"mm10 fasta already exists at {paths[3]}")
    else:
        raise ValueError("dataset must be either 'SYN' or 'GEN'.")

    return merged_tbl


import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from typing import List, Tuple, Union, Any, Mapping
import seqpro as sp

def tfbs_encode(
    seqs: List[str],
    patterns: Mapping[str, Union[str, List[str]]],
    include_rc: bool = False,
    affinty_dicts: Mapping[str, Mapping[str, float]] = None,
    affinity_shift: float = 0,
    max_length: int = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Encode sequences with known TFBS core patterns as presence/absence tensors.

    Takes in a list of N sequences and a dictionary of M TFBS core patterns. This
    dictionary can be a mapping of a single TF to TFBS or a list of TFBS per
    TF. Note that these patterns are assumed to be essential for binding of the
    TF and do not include flanking sequences that modulate binding affinity.
    
    Each TFBS is then scanned across each sequence and the presence of the
    TFBS is encoded as a 1 in a single channel tensor. If a TF has multiple
    TFBS, these are summed into a single channel.

    Optionally, the reverse complement of the patterns can be included in the
    encoding as a separate channel.
    
    Affinity can be included as a weight to the encoding. This is done by supplying a dictionary of
    every possible combination of flanks to the core TFBS as keys and the affinity of that
    TFBS as values. The affinity is then multiplied with the presence/absence encoding.

    Parameters
    ----------
    seqs : List[str]
        List of sequences to encode.
    patterns : Mapping[str, Union[str, List[str]]]
        Dictionary of M TFs to encode as keys and their core TFBS patterns as values.
        Can be a single TFBS (str) or a list of TFBS per TF (List[str]).
    include_rc : bool, optional
        Whether to add the reverse complement of the patterns for a TF as a separate channel.
        Appends "R" to the TF name in output list of TFs.
        Defaults to False.
    affinty_dicts : Mapping[str, Mapping[str, float]], optional
        Dictionary of M TFs as keys that match the patterns. Values are dictionaries
        of every possible combination of flanks to the core TFBS as keys and the affinity
        of that TFBS as values. The affinity is then multiplied with the presence/absence encoding.
        Defaults to None.
    affinity_shift : float, optional
        Shift to apply to the affinity vectors. This is determined by where the core of the TFBS
        is relative to the flanking sequences. e.g. if the TFBS core is 4bp and the flanking sequences are 2bp
        on each side, the shift would be 2. Defaults to 0.
    max_length : int, optional
        Maximum length to pad the sequences to. Only needed if the sequences are of varying lengths.
        Defaults to None.
        
    Returns
    -------
    Tuple[torch.Tensor, List[str]]
        Tuple of the encoded sequences as a tensor and the list of TFs encoded in the same order
        as the channels of the tensor.
        Tensor shape is (N, M, L) where N is the number of sequences, M is the number of TFs
        and L is the length of the sequence.
        The list of TFs then corresponds to the ordering of the M channels in the tensor.
    """
    # Check inputs
    seqs = seqs.copy()
    patterns = patterns.copy()
    
    # Precompute affinity vectors for all TFBS
    if affinty_dicts:
        if verbose:
            print("Precomputing affinity vectors for all TFBS")
        affinity_vectors = {}
        for tf in list(patterns.keys()):
            aff_dict = affinty_dicts[tf]
            aff_len = len(list(aff_dict.keys())[0])
            affs = []
            for seq in seqs:
                aff_list = []
                for pos in range(len(seq)-aff_len+1):
                    aff_list.append(aff_dict[seq[pos:pos+aff_len]])
                aff_list = np.array(aff_list)
                aff_list = np.pad(aff_list, (affinity_shift, 0), 'constant', constant_values=0)
                aff_list = np.pad(aff_list, (0, len(seq)-len(aff_list)), 'constant', constant_values=0)
                affs.append(aff_list)
            affs = np.array(affs)
            affinity_vectors[tf] = torch.tensor(affs, dtype=torch.float32)

    # Prepare patterns -- cast to seqs and include reverse complement if specified
    for tf in list(patterns.keys()):
        patterns[tf] = sp.cast_seqs(patterns[tf])
        if include_rc:
            patterns[tf+"R"] = sp.reverse_complement(patterns[tf], alphabet=sp.DNA, length_axis=-1)

    # Prepare seqs -- pad, ohe and convert to tensor
    pad_seqs = sp.pad_seqs(seqs, pad="right", pad_value="N", length=max_length)
    ohe_seqs = sp.ohe(pad_seqs, alphabet=sp.DNA)
    ohe_seqs = torch.tensor(ohe_seqs, dtype=torch.float32).permute(0, 2, 1)
    ohe_in_seqs = []

    # Prepare encoding -- ohe patterns and convolve
    for tf in patterns:
        if verbose:
            print("Encoding", tf)
        ohe_tf = torch.tensor(sp.ohe(patterns[tf], alphabet=sp.DNA), dtype=torch.float32)
        print(ohe_tf)
        if len(ohe_tf.shape) == 2:
            ohe_tf = ohe_tf.unsqueeze(0)
        ohe_tf = ohe_tf.permute(0, 2, 1)
        ohe_tf_in_seq = []
        for ohe_pattern in ohe_tf:
            pattern_len = ohe_pattern.shape[-1]
            ohe_pattern = ohe_pattern.unsqueeze(0)
            ohe_pattern_in_seq = F.conv1d(ohe_seqs, ohe_pattern, stride=1)
            print(ohe_pattern_in_seq.max())
            print(pattern_len)
            ohe_pattern_in_seq = F.pad(ohe_pattern_in_seq, (0, len(ohe_pattern[0]) - 1))
            ohe_pattern_in_seq = torch.floor(ohe_pattern_in_seq / pattern_len)
            ohe_tf_in_seq.append(ohe_pattern_in_seq)
        ohe_tf_in_seq = torch.cat(ohe_tf_in_seq, dim=1)
        ohe_tf_in_seq = torch.sum(ohe_tf_in_seq, dim=1, keepdim=True)
        if affinty_dicts:
            aff_vec = affinity_vectors[tf.replace("R", "")]
            aff_vec = aff_vec.unsqueeze(1)
            ohe_tf_in_seq = ohe_tf_in_seq * aff_vec
        ohe_in_seqs.append(ohe_tf_in_seq)
    ohe_in_seqs = torch.cat(ohe_in_seqs, dim=1)

    return ohe_in_seqs, list(patterns.keys())


class ResidualBind(nn.Module):

    def __init__(
        self,
        input_len,
        output_dim,
        input_channels=4,
        conv_channels=96,
        conv_kernel=11,
        conv_stride=1,
        conv_dilation=1,
        conv_padding="valid",
        conv_bias=False,
        conv_norm_type="batchnorm",
        conv_activation="relu",
        conv_dropout_rate=0.1,
        conv_order="conv-norm-act-dropout",
        num_residual_blocks=3,
        residual_channels=96,
        residual_kernel=3,
        residual_stride=1,
        residual_dilation_base=2,
        residual_biases=False,
        residual_activation="relu",
        residual_norm_type="batchnorm",
        residual_dropout_rates=0.1,
        residual_order="conv-norm-act-dropout",
        avg_pool_kernel=10,
        avg_pool_dropout_rate=0.2,
        dense_hidden_dims=[256],
        dense_biases=False,
        dense_activation="relu",
        dense_norm_type="batchnorm",
        dense_dropout_rates=0.5,
        dense_order="linear-norm-act-dropout",
    ):
        super(ResidualBind, self).__init__()

        # Set the attributes
        self.input_len = input_len
        self.output_dim = output_dim

        self.conv1d_block = sm.Conv1DBlock(
            input_len=input_len,
            input_channels=input_channels,
            output_channels=conv_channels,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            conv_dilation=conv_dilation,
            conv_padding=conv_padding,
            conv_bias=conv_bias,
            activation=conv_activation,
            pool_type=None,
            dropout_rate=conv_dropout_rate,
            norm_type=conv_norm_type,
            order=conv_order,
        )
        self.res_tower = sm.layers.Residual(sm.Tower(
            input_size=(self.conv1d_block.output_size),
            block=sm.Conv1DBlock,
            repeats=num_residual_blocks,
            static_block_args={
                'input_len': self.conv1d_block.output_size[-1], 
                'output_channels': residual_channels,
                'conv_kernel': residual_kernel,
                'conv_stride': residual_stride,
                'conv_padding': 'same',
                'conv_bias': residual_biases,
                'activation': residual_activation,
                'pool_type': None,
                'norm_type': residual_norm_type,
                'dropout_rate': residual_dropout_rates,
                'order': residual_order,
            },
            dynamic_block_args={
                'input_channels': [conv_channels] + [residual_channels] * (num_residual_blocks - 1),
                'conv_dilation': [residual_dilation_base**i for i in range(num_residual_blocks)],
            }
        ))
        self.average_pool = nn.AvgPool1d(kernel_size=avg_pool_kernel, stride=1, padding=0)
        self.average_pool_dropout = nn.Dropout(p=avg_pool_dropout_rate)
        self.flatten = nn.Flatten()
        self.flatten_dim = self.res_tower.wrapped.output_size[-2] * (self.res_tower.wrapped.output_size[-1]-avg_pool_kernel+1)
        self.dense_tower = sm.Tower(
            input_size=self.flatten_dim,
            block=sm.DenseBlock,
            repeats=len(dense_hidden_dims)+1,
            static_block_args={
                'activation': dense_activation,
                'bias': dense_biases,
                'norm_type': dense_norm_type,
                'dropout_rate': dense_dropout_rates,
                'order': dense_order,
            },
            dynamic_block_args={
                'input_dim': [self.flatten_dim] + dense_hidden_dims,
                'output_dim': dense_hidden_dims+[1], 
                'dropout_rate': [0.5, None], 
                'order': ['linear-norm-act-dropout', 'linear']},
        )


    def forward(self, x):
        x = self.conv1d_block(x)
        x = self.res_tower(x)
        x = self.average_pool(x)
        x = self.average_pool_dropout(x)
        x = self.flatten(x)
        x = self.dense_tower(x)
        return x