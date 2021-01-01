"""
This file contains the implementation of the scoring metric so that you can use it locally
or reimplement it in other languages. See the problem description for an explanation of
the metric.
"""

import numpy as np
from loguru import logger
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


class Deid2Metric:
    def __init__(self):
        self.threshold = 0.05
        self.misleading_presence_penalty = 0.2
        self.bias_penalty = 0.25
        self.allowable_raw_bias = 500

    @staticmethod
    def _normalize(arr):
        """ Turn raw counts into frequencies but handle all-zero arrays gracefully """
        if (arr > 0).any():
            return arr / arr.sum()
        return arr

    def _zero_below_threshold(self, arr):
        """ Take any entries that are below the threshold we care about and zero them out """
        return np.where(self._normalize(arr) >= self.threshold, arr, 0)

    @logger.catch
    def _penalty_components(self, actual, predicted):
        """ Score one row of counts for a particular (neighborhood, year, month) """
        if (actual == predicted).all():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # get the overall penalty for bias
        bias_mask = np.abs(actual.sum() - predicted.sum()) > self.allowable_raw_bias
        bias_penalty = self.bias_penalty if bias_mask.any() else 0

        # zero out entries below the threshold
        gt = self._zero_below_threshold(actual).ravel()
        dp = self._zero_below_threshold(predicted).ravel()
        """
        # get the base Jensen Shannon distance; add a tiny bit of weight to each bin in order
        # to avoid all-zero arrays (and thus NaNs) without unduly influencing the distribution
        # induced by normalizing the arrays (dividing by sums); use base 2 to get a proper
        # spread of scores between [0, 1] instead of default base e which is more like [0, 0.83]
        #
        # ref: https://docs.scipy.org/doc/scipy-1.5.2/reference/generated/scipy.spatial.distance.jensenshannon.html
        jsd = jensenshannon(gt + 1e-9, dp + 1e-9, base=2)
        if np.isnan(jsd):
            # the following are the detailed operations of jensenshannon; it will output nan due to precision loss
            # computing kl seems more robust
            p, q = gt + 1e-9, dp + 1e-9
            p = np.asarray(p)
            q = np.asarray(q)
            p = p / np.sum(p, axis=0)
            q = q / np.sum(q, axis=0)
            m = (p + q) / 2.0
            left = rel_entr(p, m)
            right = rel_entr(q, m)
            js = np.sum(left, axis=0) + np.sum(right, axis=0)
            js /= np.log(2)
            print('err')
            raise ValueError(gt, dp)
        """
        # get the overall penalty for the presence of misleading counts
        misleading_presence_mask = (gt == 0) & (dp > 0)
        misleading_presence_penalty = (misleading_presence_mask.sum() * self.misleading_presence_penalty)

        # newly added
        p, q = gt + 1e-9, dp + 1e-9
        p, q = p / p.sum(), q / q.sum()
        kl1, kl2 = entropy(p, (p + q) / 2, base=2), entropy(q, (p + q) / 2, base=2)
        if kl1 + kl2 < 0:
            jsd = 0
        else:
            jsd = ((kl1 + kl2) / 2) ** 0.5
        # assert jsd == ((kl1 + kl2) / 2) ** 0.5 # there is slight difference

        jaccard_nom = (gt > 0) & (dp > 0)
        jaccard_nom = jaccard_nom.sum()
        jaccard_denom = (gt > 0).sum() + (dp > 0).sum() - jaccard_nom
        if jaccard_denom == 0:
            jaccard_dist = 0
        else:
            jaccard_dist = jaccard_nom / jaccard_denom

        return jsd, misleading_presence_penalty, bias_penalty, kl1, kl2, jaccard_dist

    def _raw_row_scores(self, actual, predicted, actual_sampled):
        n_rows, _n_incidents = actual.shape
        raw_penalties = np.zeros((n_rows, 6), dtype=np.float)
        for i in range(n_rows):
            components_i = self._penalty_components(actual[i, :], predicted[i, :])
            raw_penalties[i] = components_i

        print(
            f'{np.sum(predicted):.1f}, {np.sum(raw_penalties[:, 0]):.1f}, {np.sum(raw_penalties[:, 1]):.1f}, {np.sum(raw_penalties[:, 2]):.1f}, {np.sum(raw_penalties[:, 3]):.1f}, {np.sum(raw_penalties[:, 4]):.1f}',
            end=', '
        )

        # print row specific scores
        # csv_res = np.zeros((3336, 6))
        # for i in range(n_rows):
        #     csv_res[i] = i + 1, np.sum(predicted[i]), np.count_nonzero(self._zero_below_threshold(predicted[i, :])), raw_penalties[i][0], raw_penalties[i][1], raw_penalties[i][2]
        # np.savetxt("csv_res.csv", csv_res, delimiter=",", fmt='%.1f')

        # print row details todo: use the same randomness
        # csv_res = np.zeros((3336 * 3, 174))
        # for i in range(n_rows):
        #     row = actual[i]
        #     order_index = np.argsort(row)[::-1]
        #     csv_res[3 * i: 3 * i + 3] = actual[i, order_index], actual_sampled[i, order_index], predicted[i, order_index]
        # np.savetxt("csv_res_detailed.csv", csv_res, delimiter=",", fmt='%.1f')

        penalties = np.sum(raw_penalties[:, : 3], axis=1)
        raw_scores = np.ones_like(penalties) - penalties

        # figure(num=None, figsize=(10, 10), dpi=180, facecolor='w', edgecolor='k')
        # sums = np.sum(actual, axis=1)
        # plt.scatter(sums, 1 - raw_penalties[:, 0], s=8, alpha=0.5)
        # plt.scatter(raw_penalties[:, 5], raw_penalties[:, 0], s=8, alpha=0.5)
        # plt.scatter(raw_penalties[:, 5], raw_penalties[:, 1], s=8, alpha=0.5)

        np.set_printoptions(precision=1)
        np.set_printoptions(suppress=True)
        # min_jsd_index = np.argmax(raw_penalties[:, 0])
        # print(f'{min_jsd_index}, ', ', '.join('{0:0.1f}'.format(x) for x in raw_penalties[min_jsd_index]))
        # row = actual[min_jsd_index, :]
        # row_sampled = actual_sampled[min_jsd_index, :]
        # new_index = np.argsort(row)[::-1]
        # print(', '.join('{0:0.1f}'.format(x) for x in row[new_index]))
        # print(', '.join('{0:0.1f}'.format(x) for x in row_sampled[new_index]))
        # print(', '.join('{0:0.1f}'.format(x) for x in predicted[min_jsd_index, new_index]))

        # min_score = np.argmin(raw_scores)
        # print(f'{min_score}, {raw_scores[min_score]:.1f}', end=', ')
        # print(', '.join('{0:0.1f}'.format(x) for x in raw_penalties[min_score]))
        # print(', '.join('{0:0.1f}'.format(x) for x in actual[min_score, :]))
        # print(', '.join('{0:0.1f}'.format(x) for x in self._zero_below_threshold(actual[min_score, :]).ravel()))
        # print(', '.join('{0:0.1f}'.format(x) for x in predicted[min_score, :]))
        # print(', '.join('{0:0.1f}'.format(x) for x in self._zero_below_threshold(predicted[min_score, :]).ravel()))
        # print(f'{raw_penalties[min_score]}\n {actual[min_score, :]}\n {self._zero_below_threshold(actual[min_score, :]).ravel()}\n {predicted[min_score, :]}\n {self._zero_below_threshold(predicted[min_score, :]).ravel()}')
        return raw_scores

    def score(self, actual, predicted, actual_sampled=None, return_individual_scores=False):
        # make sure the submitted values are proper
        assert np.isfinite(predicted).all()
        assert (predicted >= 0).all()

        # get all of the individual scores
        raw_scores = self._raw_row_scores(actual, predicted, actual_sampled)

        # clip all the scores to [0, 1]
        scores = np.clip(raw_scores, a_min=0.0, a_max=1.0)

        # sum up the scores - a perfect score would be the length of the submission format
        overall_score = np.sum(scores)

        # in some uses (like visualization), it's useful to get the individual scores out too
        if return_individual_scores:
            return overall_score, scores

        return overall_score

    def normalized_by_row(self, overall_score, actual):
        normalized_score = overall_score / actual.shape[0]
        return normalized_score
