from problem_spec.EventAnalysis import EventAnalysis
import argparse

def main():
    problem = EventAnalysis(args.Delta, args.alpha)
    problem.calculate_AEMD(target_dp_data_label=args.dataset, compare_strategy=args.strategy)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AEMD')
    parser.add_argument('--dataset', type=str, default='mid', choices=['poor', 'mid'])
    parser.add_argument('--strategy', type=str, default='month_only', choices=['month_only', 'month_incident_dummy',
                                                                               'month_only_V2', 'month_incident_dummy_V2',
                                                                               'month_only_flow', 'month_incident_dummy_flow',
                                                                               'month_only_flow_V2', 'month_incident_dummy_flow_V2',
                                                                               'three_way_margin_flow'])
    parser.add_argument('--Delta', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=1)
    args = parser.parse_args()
    main()
