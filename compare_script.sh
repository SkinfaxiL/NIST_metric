#!/bin/bash

nohup python3 compare.py --dp_data ./data/Event/privatized-mediocre-quality.csv &
nohup python3 compare.py --dp_data ./data/Event/privatized-very-poor-quality.csv &

# DP syn data scores

nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-0.1-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-0.25-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-0.5-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-1.0-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-1.0-4.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-2.0-2.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-2.0-4.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-10.0-9.pkl --is_dp_data_pickle True &
nohup python3 compare.py --dp_data ./data/event_dpsyn/2019-dpsyn-100-20.pkl --is_dp_data_pickle True &