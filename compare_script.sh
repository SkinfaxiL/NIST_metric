#!/bin/bash

nohup python3 compare.py --dp_data ./data/Event/privatized-mediocre-quality.csv &
nohup python3 compare.py --dp_data ./data/Event/privatized-very-poor-quality.csv &

## sprint1 submission
#nohup python3 compare.py --dp_data ./data/sprint1/eps1.csv &
#nohup python3 compare.py --dp_data ./data/sprint1/eps2.csv &
#nohup python3 compare.py --dp_data ./data/sprint1/eps10.csv &

# sprint1 laplace
nohup python3 compare.py --dp_data ./data/sprint1/lap_eps0.1.csv &
nohup python3 compare.py --dp_data ./data/sprint1/lap_eps0.25.csv &
nohup python3 compare.py --dp_data ./data/sprint1/lap_eps0.5.csv &
nohup python3 compare.py --dp_data ./data/sprint1/lap_eps1.csv &
nohup python3 compare.py --dp_data ./data/sprint1/lap_eps2.csv &
nohup python3 compare.py --dp_data ./data/sprint1/lap_eps4.csv &
nohup python3 compare.py --dp_data ./data/sprint1/lap_eps10.csv &


# DP syn data scores

nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-0.1-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-0.25-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-0.5-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-1.0-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-2.0-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-4.0-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-10.0-9.pkl --is_dp_data_pickle True &
#nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-100-20.pkl --is_dp_data_pickle True &

# sprint1 sampled
nohup python3 compare.py --dp_data ./data/sprint1/0.01_sampling.csv &
nohup python3 compare.py --dp_data ./data/sprint1/0.05_sampling.csv &
nohup python3 compare.py --dp_data ./data/sprint1/0.1_sampling.csv &
nohup python3 compare.py --dp_data ./data/sprint1/0.25_sampling.csv &
nohup python3 compare.py --dp_data ./data/sprint1/0.5_sampling.csv &
#nohup python3 compare.py --dp_data ./data/sprint1/0.8_sampling.csv &
