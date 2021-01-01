import numpy as np
import pandas as pd


class RangeQuery():
    def __init__(self):
        self.neighborhoods = 278
        self.incident_types = 174
        self.months = 12

    def score(self, ground_truth, dp_data,  month_range, neighborhood_range, incident_types_range, num_tails=300):
        total_diff = 0
        relative_err = 0
        min_threshold = 10
        for i in range(num_tails):
            # todo: random sample neighborhoods, incident types, months
            sampled_neighbors = np.random.choice(self.neighborhoods, neighborhood_range, replace=False)
            sampled_incidents = np.random.choice(self.incident_types, incident_types_range, replace=False)
            sampled_incidents = [str(i) for i in sampled_incidents]
            sampled_month = np.random.choice(self.months, month_range, replace=False)
            sampled_month += 1
            sampled_gt = ground_truth.loc[pd.IndexSlice[sampled_neighbors, sampled_month], sampled_incidents]
            sampled_dp = dp_data.loc[pd.IndexSlice[sampled_neighbors, sampled_month], sampled_incidents]
            gt_count = np.sum(sampled_gt.to_numpy())
            dp_count = np.sum(sampled_dp.to_numpy())
            total_diff += np.abs(gt_count - dp_count)
            relative_err += np.abs(gt_count - dp_count) / max(gt_count, min_threshold)
        return total_diff / num_tails, relative_err / num_tails
