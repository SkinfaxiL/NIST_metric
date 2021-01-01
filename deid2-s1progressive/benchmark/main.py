import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

ROOT_DIRECTORY = Path("codeexecution")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

DEFAULT_SUBMISSION_FORMAT = DATA_DIRECTORY / "submission_format.csv"
DEFAULT_PARAMS = DATA_DIRECTORY / "parameters.json"
DEFAULT_OUTPUT = ROOT_DIRECTORY / "submission.csv"
PRIVATE_INCIDENTS = DATA_DIRECTORY / "incidents.csv"
PUBLIC_INCIDENTS = ROOT_DIRECTORY / "pub_incidents.csv"


#############################################
# Part 1: Basic DP Mechanisms               #
#############################################
def naively_add_laplace_noise(arr, scale: float, seed: int = None):
    """
        Add Laplace random noise of the desired scale to the dataframe of counts.
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.laplace(scale=scale, size=arr.size).reshape(arr.shape)
    ### our change: does not enforce non-negativity and integer
    # result = np.clip(arr + noise, a_min=0, a_max=np.inf)
    result = arr + noise
    # return result.round().astype(np.int)
    return result


def em(q_ans, eps, sensitivity: int, monotonic=True) -> int:
    """
        Exponential mechanism.  
        To avoid overflow, we make sure the highest score is smaller than 25.
    """
    coeff = eps / sensitivity if monotonic else eps / 2 / sensitivity
    q_ans *= coeff
    if np.max(q_ans) > 25:
        q_ans -= np.max(q_ans) - 25
    probs = np.exp(q_ans)
    probs = probs / sum(probs) if sum(probs) > 0 else [1] * len(probs)
    probs = np.cumsum(probs)

    rand_p = np.random.rand()
    return np.searchsorted(probs, rand_p, side='left')


#############################################
# Part 2: Data Processing Functions         #
#############################################
def get_ground_truth(incident_csv: Path, submission_format: pd.DataFrame) -> np.ndarray:
    """
        Method to read the data and summarize into the table.
        Copied it from the given code.
    """
    incidents = pd.read_csv(incident_csv, index_col=0)
    incidents['n'] = 1
    counts = get_table(incidents, submission_format)
    return counts.values


def get_ground_truth_truncate(incident_csv: Path, submission_format: pd.DataFrame, truncate: int,
                              max_sensitivity: int) -> np.ndarray:
    """
        Method to read the data and summarize into the table.
        We truncate the data so that each individual contributes to at most theta records.
    """
    incidents = pd.read_csv(incident_csv, index_col=0)
    incidents['n'] = 1
    cur_incidents = truncate_df(incidents, truncate, max_sensitivity)
    counts = get_table(cur_incidents, submission_format)
    return counts.values


def truncate_df(incidents: pd.DataFrame, truncate: int, max_sensitivity: int) -> pd.DataFrame:
    """
        Method to truncate the data.
        We first group the data by each resident.
        For residents that contribute to more than theta records, we
    """
    cur_incidents = incidents.copy()
    if truncate < max_sensitivity:
        incidents_with_count = cur_incidents.groupby(['sim_resident'], sort=False).size().reset_index(name='count')
        incidents_with_count['weight'] = 1
        incidents_with_count.loc[incidents_with_count['count'] > truncate, ['weight']] /= truncate
        cur_incidents = cur_incidents.merge(incidents_with_count, on='sim_resident', how='left')
        cur_incidents['n'] = cur_incidents['n'] / cur_incidents['weight']
    return cur_incidents


def get_sum_truncate(incident_csv: Path, truncate: int, max_sensitivity: int) -> int:
    """
        Method to read the data and get the sum.
    """
    incidents = pd.read_csv(incident_csv, index_col=0)
    incidents['n'] = 1
    cur_incidents = truncate_df(incidents, truncate, max_sensitivity)
    return cur_incidents.agg({'n': ['sum']}).values[0][0]


def get_table(incidents: pd.DataFrame, submission_format: pd.DataFrame) -> pd.DataFrame:
    """
        Method to create the table.
        Copied from the given code.
    """
    counts = incidents.pivot_table(
        index=["neighborhood", "year", "month"],
        columns="incident_type",
        values="n",
        aggfunc=np.sum,
        fill_value=0,
    )

    # when you pivot, you only gets rows and columns for things that were actually there --
    # the ground truth may not have all of the neighborhoods, periods, or codes we expected to see,
    # so we'll fix that by reindexing and then filling the missing values
    epsilons = submission_format.index.levels[0]
    index_for_one_epsilon = submission_format.loc[epsilons[0]].index
    columns = submission_format.columns.astype(counts.columns)
    counts = (counts.reindex(columns=columns, index=index_for_one_epsilon).fillna(0).astype(np.int32))
    return counts


#############################################
# Part 3: Our Methods                       #
#############################################
def answer_adap(eps: float, submission_format, max_sensitivity: int) -> np.ndarray:
    """
        Adaptively choose which method to use based on two estimated errors
    """
    # privacy budget allocation
    l1_eps = 0.01 * eps
    priv_noise_eps = 0.01 * eps
    em_eps = 0.01 * eps
    ret_eps = eps - l1_eps - em_eps - priv_noise_eps

    # get the l1 distance between the private data and the public data
    from numpy import linalg as LA
    truth = get_ground_truth(PRIVATE_INCIDENTS, submission_format)
    pub = get_ground_truth(PUBLIC_INCIDENTS, submission_format).astype(np.float)
    pub_norm = 0
    for row_i, row in enumerate(truth):
        pub_row = pub[row_i]
        if np.sum(pub_row) > 0:
            # each public row is normalized to have the same sum as the private row
            # we have proved this has the sensitivity of 2 * max_sensitivity
            pub_norm += LA.norm(row - pub_row * np.sum(row) / np.sum(pub_row), 1)
        else:
            pub_norm += LA.norm(row, 1)
    est_pub_norm = naively_add_laplace_noise(pub_norm, 2 * max_sensitivity / l1_eps)

    # get the expected noise error of using dp.
    # the noise error can be directly obtained; but the factor needs to be estimated
    theta = em_theta(em_eps, ret_eps, max_sensitivity)
    original_sum = np.sum(truth)
    est_original_sum = naively_add_laplace_noise(original_sum, max_sensitivity / (0.5 * priv_noise_eps))
    truncated_sum = get_sum_truncate(PRIVATE_INCIDENTS, theta, max_sensitivity)
    est_truncated_sum = naively_add_laplace_noise(truncated_sum, max_sensitivity / (0.5 * priv_noise_eps))
    truncate_factor = est_original_sum / est_truncated_sum
    est_noise_error = 0.14 * truncate_factor * theta / ret_eps * 3336 * 174

    # decide which method to use based on the comparison between the two errors
    if est_noise_error <= est_pub_norm:
        return answer_dp(ret_eps, submission_format, max_sensitivity, theta)
    else:
        return answer_pub(ret_eps, submission_format, max_sensitivity)


def em_theta(em_eps: float, est_eps: float, max_sensitivity: int) -> int:
    """
        Method to calculate the quality function (sum after truncation minus noise error) and run em.
    """
    q_len = max(10, int(max_sensitivity / 2))
    em_sensitivity = q_len + 1
    q_vec = np.zeros(q_len)
    for theta in range(1, 1 + q_len):
        q_vec[theta - 1] = get_sum_truncate(PRIVATE_INCIDENTS, theta, max_sensitivity) - theta / est_eps * 3336 * 174

    truncate = em(q_vec, em_eps, em_sensitivity, True) + 1
    return truncate


def answer_dp(eps: float, submission_format: pd.DataFrame, max_sensitivity: int, truncate: int = 0) -> np.ndarray:
    """
        Method to use dp to answer.
    """
    # first estimate theta, the truncation threshold
    if truncate == 0:
        em_eps = min(0.01 * eps, 0.01)
        est_eps = eps - em_eps
        truncate = em_theta(em_eps, est_eps, max_sensitivity)
    else:
        # when this function is called from answer_adap, we have already obtained theta
        em_eps = 0

    # obtain the sum for the rescaling step
    if truncate > min(8, max_sensitivity * 2 / 3):
        # when theta is large, we ignore the rescaling step
        total_eps = 0
        est_total = -1
    else:
        total_eps = min(0.01 * eps, 0.01)
        true_total = get_sum_truncate(PRIVATE_INCIDENTS, truncate, max_sensitivity)
        est_total = naively_add_laplace_noise(true_total, max_sensitivity / total_eps)

    # obtain the ground truth and add laplace noise
    lap_eps = eps - em_eps - total_eps
    truth = get_ground_truth_truncate(PRIVATE_INCIDENTS, submission_format, truncate, max_sensitivity)
    ret = naively_add_laplace_noise(truth, truncate / lap_eps)
    ret = np.clip(ret, a_min=0, a_max=np.inf)

    # the optional rescaling step
    if est_total > 0:
        ret *= est_total / np.sum(ret)
    return ret


def answer_pub(eps: float, submission_format: pd.DataFrame, max_sensitivity: int) -> np.ndarray:
    """
        Method to use public data to answer.
    """
    truncate = int(math.ceil(min(10, 600 * eps)))
    num_categories = (278, 12, 174)
    truth = get_ground_truth_truncate(PRIVATE_INCIDENTS, submission_format, truncate, max_sensitivity)
    truth_3d = np.reshape(truth, num_categories)

    neighborhood_month = np.sum(truth_3d, axis=2)
    est_neighborhood_month = naively_add_laplace_noise(neighborhood_month, truncate / eps)
    est_neighborhood_month = np.clip(est_neighborhood_month, a_min=0, a_max=np.inf)

    pub = get_ground_truth(PUBLIC_INCIDENTS, submission_format).astype(np.float)
    for row_i, row in enumerate(est_neighborhood_month):
        pub_row = pub[row_i]
        if np.sum(pub_row) > 0:
            pub_row *= np.sum(row) / np.sum(pub_row)

    return pub


def main(
        submission_format: Path = DEFAULT_SUBMISSION_FORMAT,
        output_file: Optional[Path] = DEFAULT_OUTPUT,
        params_file: Path = DEFAULT_PARAMS,
):
    logger.info("loading parameters")
    params = json.loads(params_file.read_text())

    # read in the submission format
    logger.info(f"reading submission format from {submission_format} ...")
    submission_format = pd.read_csv(
        submission_format, index_col=["epsilon", "neighborhood", "year", "month"]
    )
    logger.info(f"read dataframe with {len(submission_format):,} rows")

    logger.info("counting up incidents by (neighborhood, year, month)")

    logger.info(f"privatizing each set of {len(submission_format):,} counts...")
    submission = submission_format.copy()
    with tqdm(total=len(params["runs"])) as pbar:
        for run in params["runs"]:
            ###############################################################################
            # NOTE: THIS IS THE DIFFERENTIAL-PRIVACY SENSITIVE PORTION                    #
            # We do it table by table, instead of row by row in the given code            #
            ###############################################################################
            eps = run["epsilon"]
            max_sensitivity = run["max_records_per_individual"]

            if eps < 0.25:
                privatized_counts = answer_pub(eps, submission_format, max_sensitivity)
            elif eps < 5:
                privatized_counts = answer_adap(eps, submission_format, max_sensitivity)
            else:
                privatized_counts = answer_dp(eps, submission_format, max_sensitivity)

            # put these counts in the submission dataframe
            privatized_counts = np.clip(privatized_counts, a_min=0, a_max=np.inf)
            submission.loc[eps] = privatized_counts.round().astype(np.int)

            # update the progress bar
            pbar.update(1)

    if output_file is not None:
        logger.info(f"writing {len(submission_format):,} rows out to {output_file}")
        submission.to_csv(output_file, index=True)

    return submission_format


if __name__ == "__main__":
    typer.run(main)
