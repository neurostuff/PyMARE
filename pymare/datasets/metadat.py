"""Datasets from metadat."""

import json
import os.path as op

import pandas as pd

from pymare.utils import get_resource_path


def michael2013():
    """Load a dataset of studies on the persuasive power of a brain image.

    This dataset was published in :footcite:t:`michael2013non`,
    and was curated in metadat :footcite:p:`white2022metadat`.

    Returns
    -------
    df : :obj:`~pandas.DataFrame`
        A dataframe with the following columns:

        - ``"Study"``: the study name
        - ``"No_brain_n"``: the sample size for no-brain-image condition
        - ``"No_brain_m"``: mean agreement rating for no-brain-image condition
        - ``"No_brain_s"``: standard deviation of agreement rating for no-brain-image condition
        - ``"Brain_n"``: the sample size for brain-image condition
        - ``"Brain_m"``: mean agreement rating for brain-image condition
        - ``"Brain_s"``: standard deviation of agreement rating for brain-image condition
        - ``"Included_Critique"``: whether a critique was included in the study or not
        - ``"Medium"``: the medium of the study
        - ``"Compensation"``: notes on the compensation of the study
        - ``"Participant_Pool"``: notes on where participants were recruited
        - ``"yi"``: Raw mean difference, calculated as Brain_m - No_brain_m
        - ``"vi"``: Corresponding sampling variance

    metadata : :obj:`dict`
        A dictionary with metadata about the columns in the dataset.

    Notes
    -----
    For more information about this dataset, see metadat's documentation:
    https://wviechtb.github.io/metadat/reference/dat.michael2013.html

    References
    ----------
    .. footbibliography::
    """
    dataset_dir = op.join(get_resource_path(), "datasets")
    tsv_file = op.join(dataset_dir, "michael2013.tsv")
    json_file = op.join(dataset_dir, "michael2013.json")
    df = pd.read_table(tsv_file)
    with open(json_file, "r") as fo:
        metadata = json.load(fo)

    return df, metadata
