from problem_spec.EventAnalysis import EventAnalysis

def main():
    problem = EventAnalysis()
    problem.calculate_AEMD(target_dp_data_label='mid', compare_strategy='month_only')



if __name__ == "__main__":
    main()
