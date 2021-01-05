from problem_spec.Event3way import Event3way
from problem_spec.Event2way_month_incident import Event2way_month_incident
from problem_spec.Event2way_month_neighbor import Event2way_month_neighbor
from problem_spec.Event2way_neighbor_incident import Event2way_neighbor_incident
from problem_spec.Event1way_month import Event1way_month
from problem_spec.Event1way_neighbor import Event1way_neighbor
from problem_spec.Event1way_incident_type import Event1way_incident_type
from problem_spec.piechart_metric import Deid2Metric
from problem_spec.range_query import RangeQuery
from problem_spec.Jaccard import Jaccard
from problem_spec.Jaccard2 import Jaccard2
import argparse
import pandas as pd
import pickle
import numpy as np
from copy import deepcopy
import os
import json


ground_truth_file = './data/Event/ground_truth.csv'
read_cols = ["neighborhood", "month"] + [str(i) for i in range(174)]
AEMD_metrics = {
                '1-way(month)': Event1way_month,
                '1-way(neighbor)': Event1way_neighbor,
                '1-way(incident)': Event1way_incident_type,
                '2-way(month-incident)': Event2way_month_incident,
                '2-way(month-neighborhood)': Event2way_month_neighbor,
                '2-way(neighborhood-incident)': Event2way_neighbor_incident,
                '3-way': Event3way
                }

def load_data(file_path, is_pickle=False, formate=None):
    if not is_pickle:
        data = pd.read_csv(file_path, index_col=["neighborhood", "month"], usecols=read_cols)
    else:
        data = pickle.load(open(file_path, 'rb'))
        data = np.rint(data)
        data = pd.DataFrame(data=data.values, index=formate.index, columns=formate.columns)
    return data

def save_json(result, file_name, results_dir='./result'):
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    with open(os.path.join(results_dir, file_name), 'w') as f:
        json.dump(result, f, indent=2)

def main():
    update_str = ''
    # load truth data and private data
    ground_truth = load_data(ground_truth_file)
    dp_syn = load_data(args.dp_data, args.is_dp_data_pickle, deepcopy(ground_truth))
    scores = {}
    scores['file'] = args.dp_data
    scores['Delta'] = args.Delta
    # compute pie chart score
    pie_chart_scorer = Deid2Metric()
    pie_chart_overall_score, row_scores = pie_chart_scorer.score(ground_truth.values, dp_syn.values,
                                                                 return_individual_scores=True)
    reversed_score = np.sum(1 - row_scores) / ground_truth.values.shape[0]
    normalized_score = pie_chart_scorer.normalized_by_row(pie_chart_overall_score, ground_truth.values)
    print("pie chart score:", pie_chart_overall_score, normalized_score, reversed_score)
    scores['pie_chart'] = [pie_chart_overall_score, normalized_score, reversed_score]
    update_str += '  &' + str(round(scores['pie_chart'][-1], 4))
    print("==" * 20)

    # compute rang query score
    rq_scorer = RangeQuery()
    raw_scores, relative_score = rq_scorer.score(ground_truth, dp_syn, month_range=4,
                                 neighborhood_range=int(278*0.3), incident_types_range=int(174*0.3))
    print("range query score:", raw_scores, relative_score)
    scores['range_query'] = [raw_scores, relative_score]
    update_str += '  &' + str(round(scores['range_query'][-1], 4))
    print("==" * 20)

    # compute AEMD for 7 different marginals
    for m, metric in AEMD_metrics.items():
        problem = metric(args.Delta)
        score = problem.compute_AEMC(dp_syn, ground_truth)
        normalized_score = problem.normalize(score, ground_truth)
        print('---->', m, score, normalized_score)
        scores[m] = [score, normalized_score]
        update_str += '  &' + str(round(scores[m][-1], 4))
        print("=="*20)

    scores['update_str'] = update_str
    save_json(scores, 'all_scores_Delta=' + str(args.Delta) + '_' + os.path.split(args.dp_data)[-1] + '.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AEMD')
    parser.add_argument('--dp_data', type=str, default='')
    parser.add_argument('--is_dp_data_pickle', type=bool, default=False)
    # parser.add_argument('--truth_data', type=str, default='')

    parser.add_argument('--Delta', type=float, default=2)
    parser.add_argument('--alpha', type=float, default=1)
    args = parser.parse_args()
    main()
