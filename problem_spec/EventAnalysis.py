from .Base import Base
import pandas as pd
import os
import numpy as np
from cvxopt import matrix, solvers, spmatrix
from datetime import datetime
from cvxpy import *
from ortools.graph import pywrapgraph
import math


class EventAnalysis(Base):
    def __init__(self, Delta, alpha):
        super().__init__()
        self.file_dir = './data/Event'
        self.ground_truth_file = 'ground_truth.csv'
        self.ground_truth = None
        self.dp_data_files = {"mid": 'privatized-mediocre-quality.csv',
                              "poor": 'privatized-very-poor-quality.csv'}
        read_cols = ["neighborhood", "year", "month"] + [str(i) for i in range(174)]
        self.ground_truth = pd.read_csv(os.path.join(self.file_dir, self.ground_truth_file),
                                        index_col=["neighborhood", "year", "month"],
                                        usecols=read_cols)
        self.dp_data = {}
        for label, filename in self.dp_data_files.items():
            self.dp_data[label] = pd.read_csv(os.path.join(self.file_dir, filename),
                                              index_col=["neighborhood", "year", "month"],
                                              usecols=read_cols)

        self.compare_strategy = {"month_incident_dummy": self._month_incident_dummy,
                                 "month_only": self._month_only,
                                 "month_only_V2": self._month_only_V2,
                                 "month_incident_dummy_V2": self._month_incident_dummy_V2,
                                 "month_only_flow": self._month_only_flow,
                                 "simplest": self._simplest}
        self.Delta = Delta
        self.alpha = alpha

    def calculate_AEMD(self, target_dp_data_label, compare_strategy):
        assert target_dp_data_label in self.dp_data
        assert compare_strategy in self.compare_strategy
        return self.compare_strategy[compare_strategy](target_dp_data_label)

    def _month_incident_dummy(self, target_dp_data_label):
        print("calculate _month_incident_dummy")
        num_incident_type = 174
        num_incident_type_with_dummy = 175
        months = 12
        total_cells = num_incident_type_with_dummy * 12
        num_flow_variables = (10 * 3 + 2 * 2) * num_incident_type + months * num_incident_type
        Delta = self.Delta
        alpha = self.alpha
        dp_data = self.dp_data[target_dp_data_label]
        print("parameter: Delta/alpha", Delta, alpha)

        # design x = [flows_to_next_month] + [flows_to_previous_month] + [flow_within_month]
        to_pre_base_idx = 11 * num_incident_type
        within_month_base_idx = 22 * num_incident_type
        S = np.zeros((total_cells, num_flow_variables))
        N = np.zeros((total_cells, num_flow_variables))
        for i in range(total_cells):
            if (i + 1) % num_incident_type_with_dummy == 0:
                month = int((i + 1) / num_incident_type_with_dummy)
                related_idx_start = within_month_base_idx + (month - 1) * 2 * num_incident_type
                N[i, related_idx_start + num_incident_type: related_idx_start + 2 * num_incident_type] = 1
                N[i, related_idx_start: related_idx_start + num_incident_type] = -1
                S[i, related_idx_start: related_idx_start + num_incident_type] = 0.5
            elif i < num_incident_type_with_dummy:
                N[i, [i, within_month_base_idx + i]] = 1
                N[i, [to_pre_base_idx + i, within_month_base_idx + num_incident_type + i]] = -1
                S[i, to_pre_base_idx + i] = 1
                S[i, within_month_base_idx + num_incident_type + i] = 0.5
            elif i >= 11 * num_incident_type_with_dummy:
                incident_type = i % num_incident_type_with_dummy
                within_month_start_idx = within_month_base_idx + 22 * num_incident_type
                N[i, [to_pre_base_idx + 10 * num_incident_type + incident_type, within_month_start_idx + incident_type]] = 1
                N[i, [11 * num_incident_type + incident_type - num_incident_type, within_month_start_idx + num_incident_type + incident_type]] = -1
                S[i, 11 * num_incident_type + incident_type - num_incident_type] = 1
                S[i, within_month_start_idx + num_incident_type + incident_type] = 0.5
            else:
                m = int(i / num_incident_type_with_dummy)
                incident_type = i % num_incident_type_with_dummy
                within_month_start_idx = within_month_base_idx + m * num_incident_type
                N[i, [m * num_incident_type + incident_type, to_pre_base_idx + (m - 1) * num_incident_type + incident_type]] = 1
                N[i, within_month_start_idx + incident_type] = 1
                N[i, [m * num_incident_type + incident_type - num_incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = -1
                N[i, within_month_start_idx + num_incident_type +incident_type] = 1
                S[i, [m * num_incident_type + incident_type - num_incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = 1
                S[i, within_month_start_idx + num_incident_type +incident_type] = 0.5

        A = np.zeros((4 * total_cells + num_flow_variables, num_flow_variables + 2 * total_cells))
        A[:total_cells, :num_flow_variables] = -S
        A[:total_cells, num_flow_variables:num_flow_variables + total_cells] = alpha * np.eye(total_cells)
        A[:total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[total_cells:2 * total_cells, :num_flow_variables] = -S
        A[total_cells:2 * total_cells, num_flow_variables:num_flow_variables + total_cells] = -alpha * np.eye(
            total_cells)
        A[total_cells:2 * total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[2 * total_cells:3 * total_cells, :num_flow_variables] = N
        A[2 * total_cells:3 * total_cells, num_flow_variables: num_flow_variables + total_cells] = np.eye(total_cells)
        A[3 * total_cells:4 * total_cells, :num_flow_variables] = -N
        A[3 * total_cells:4 * total_cells, num_flow_variables: num_flow_variables + total_cells] = -np.eye(total_cells)
        A[4 * total_cells:, :num_flow_variables] = np.eye(num_flow_variables)
        print("Initialize constraint matrix done")

        # distance = np.ones((num_flow_variables, 1))
        c = np.array([0] * (num_flow_variables + total_cells) + [1] * total_cells).reshape((-1, 1))
        print("Use 1 for all distances between neighbors")

        # todo: feed into solver
        start_t = datetime.now()
        total_loss = 0
        count = 0
        for truth_group, dp_group in zip(self.ground_truth.groupby(level='neighborhood'),
                                         dp_data.groupby(level='neighborhood')):
            # if count != 139:
            #     count += 1
            #     continue
            # elif count > 139:
            #     exit()
            truth_data = truth_group[1].values.astype('float')
            dp_data = dp_group[1].values.astype('float')
            # append dummy node
            truth_data = np.append(truth_data, np.zeros((months, 1)), 1)
            dp_data = np.append(dp_data, np.zeros((months, 1)), 1)
            print("max diff of columns", np.max(np.abs(np.sum(truth_data, axis=0) - np.sum(dp_data, axis=0))))
            truth_data = truth_data.flatten()
            dp_data = dp_data.flatten()
            b = np.concatenate((alpha * dp_data, -alpha * dp_data, truth_data - Delta, -truth_data - Delta,
                                np.zeros(num_flow_variables)))

            x = Variable(num_flow_variables + 2 * total_cells)
            constraints = [A @ x >= b]
            objective = Minimize(c.T @ x)
            problem = Problem(objective, constraints)
            problem.solve(verbose=False, solver=ECOS)
            print(truth_group[0], 'problem state: ', problem.status, problem.value)
            x = np.array(x.value)
            print('solution x: ', x, np.max(x))
            total_loss += problem.value
            abs_diff = np.sum(np.abs(truth_data - dp_data))
            print("abs diff v.s. AEMD:", abs_diff, problem.value)
            print("===============")
        print("total_time:", datetime.now() - start_t)
        print("total loss:", total_loss)

        return

    def _month_only(self, target_dp_data_label):
        print("calculate _month_only")
        # todo: initial distance matrix
        num_incident_type = 174
        total_cells = num_incident_type * 12
        num_flow_variables = (10 * 2 + 2) * num_incident_type
        Delta = self.Delta
        alpha = self.alpha
        dp_data = self.dp_data[target_dp_data_label]
        print("parameter: Delta/alpha", Delta, alpha)


        # design x = [flows_to_next_month] + [flows_to_previous_month]
        to_pre_base_idx = int(num_flow_variables / 2)
        S = np.zeros((total_cells, num_flow_variables))
        N = np.zeros((total_cells, num_flow_variables))
        for i in range(total_cells):
            if i < num_incident_type:
                N[i, i] = 1
                N[i, to_pre_base_idx + i] = -1
                S[i, to_pre_base_idx + i] = 1
            elif i >= 11 * num_incident_type:
                incident_type = i % num_incident_type
                N[i, to_pre_base_idx + 10 * num_incident_type + incident_type] = 1
                N[i, i - num_incident_type] = -1
                S[i, i - num_incident_type] = 1
            else:
                m = int(i / num_incident_type)
                incident_type = i % num_incident_type
                N[i, [i, to_pre_base_idx + (m - 1) * num_incident_type + incident_type]] = 1
                N[i, [i - num_incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = -1
                S[i, [i - num_incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = 1

        A = np.zeros((4 * total_cells + num_flow_variables, num_flow_variables + 2 * total_cells))
        A[:total_cells, :num_flow_variables] = -S
        A[:total_cells, num_flow_variables:num_flow_variables + total_cells] = alpha * np.eye(total_cells)
        A[:total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[total_cells:2 * total_cells, :num_flow_variables] = -S
        A[total_cells:2 * total_cells, num_flow_variables:num_flow_variables + total_cells] = -alpha * np.eye(total_cells)
        A[total_cells:2 * total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[2 * total_cells:3 * total_cells, :num_flow_variables] = N
        A[2 * total_cells:3 * total_cells, num_flow_variables: num_flow_variables + total_cells] = np.eye(total_cells)
        A[3 * total_cells:4 * total_cells, :num_flow_variables] = -N
        A[3 * total_cells:4 * total_cells, num_flow_variables: num_flow_variables + total_cells] = -np.eye(total_cells)
        A[4 * total_cells:, :num_flow_variables] = np.eye(num_flow_variables)
        print("Initialize constraint matrix done")

        # distance = np.ones((num_flow_variables, 1))
        c = np.array([0] * (num_flow_variables + total_cells) + [1] * total_cells).reshape((-1, 1))
        print("Use 1 for all distances between neighbors")


        # todo: feed into solver
        start_t = datetime.now()
        total_loss = 0
        for truth_group, dp_group in zip(self.ground_truth.groupby(level='neighborhood'), dp_data.groupby(level='neighborhood')):
            truth_data = truth_group[1].values.astype('float')
            dp_data = dp_group[1].values.astype('float')
            # normalize
            # truth_data /= np.sum(truth_data)
            # dp_data /= np.sum(dp_data)
            print("max diff of columns", np.max(np.abs(np.sum(truth_data, axis=0) - np.sum(dp_data, axis=0))) )
            truth_data = truth_data.flatten()
            dp_data = dp_data.flatten()
            b = np.concatenate((alpha * dp_data, -alpha * dp_data, truth_data - Delta, -truth_data - Delta,
                                np.zeros(num_flow_variables)))

            x = Variable(num_flow_variables + 2 * total_cells)
            constraints = [A @ x >= b]
            objective = Minimize(c.T @ x)
            problem = Problem(objective, constraints)
            problem.solve(verbose=False)
            print(truth_group[0], 'problem state: ', problem.status, problem.value)
            x = np.array(x.value)
            print('solution x: ', x, np.max(x))
            abs_diff = np.sum(np.abs(truth_data - dp_data))
            print("abs diff v.s. AEMD:", abs_diff, problem.value)
            print("===============")
            total_loss += problem.value
        print("total_time:", datetime.now() - start_t)
        print("total loss:", total_loss)
        return


    def _month_only_V2(self, target_dp_data_label):
        print("calculate _month_only_V2")
        # todo: initial distance matrix
        num_incident_type = 174
        total_cells = num_incident_type * 12
        num_flow_variables = (10 * 2 + 2) * num_incident_type
        Delta = self.Delta
        alpha = self.alpha
        dp_data = self.dp_data[target_dp_data_label]
        print("parameter: Delta/alpha", Delta, alpha)

        to_pre_base_idx = int(num_flow_variables / 2)
        S = np.zeros((total_cells, num_flow_variables))
        N = np.zeros((total_cells, num_flow_variables))
        for i in range(total_cells):
            if i < num_incident_type:
                N[i, i] = 1
                N[i, to_pre_base_idx + i] = -1
                S[i, to_pre_base_idx + i] = 1
            elif i >= 11 * num_incident_type:
                incident_type = i % num_incident_type
                N[i, to_pre_base_idx + 10 * num_incident_type + incident_type] = 1
                N[i, i - num_incident_type] = -1
                S[i, i - num_incident_type] = 1
            else:
                m = int(i / num_incident_type)
                incident_type = i % num_incident_type
                N[i, [i, to_pre_base_idx + (m - 1) * num_incident_type + incident_type]] = 1
                N[i, [i - num_incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = -1
                S[i, [i - num_incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = 1

        A = np.zeros((3 * total_cells + num_flow_variables, num_flow_variables + 2 * total_cells))
        A[:total_cells, :num_flow_variables] = -S
        A[:total_cells, num_flow_variables:num_flow_variables + total_cells] = alpha * np.eye(total_cells)
        A[:total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[total_cells:2 * total_cells, :num_flow_variables] = -S
        A[total_cells:2 * total_cells, num_flow_variables:num_flow_variables + total_cells] = -alpha * np.eye(total_cells)
        A[total_cells:2 * total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[2 * total_cells:3 * total_cells, :num_flow_variables] = -S
        A[2 * total_cells:3 * total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[3 * total_cells:, :num_flow_variables] = np.eye(num_flow_variables)
        print("Initialize constraint matrix done")

        F = np.zeros((total_cells, num_flow_variables + 2 * total_cells))
        F[:, :num_flow_variables] = N
        F[:, num_flow_variables:num_flow_variables + total_cells ] = np.eye(total_cells)
        # distance = np.ones((num_flow_variables, 1))
        c = np.array([0] * (num_flow_variables + total_cells) + [1] * total_cells).reshape((-1, 1))
        print("Use 1 for all distances between neighbors")


        # todo: feed into solver
        start_t = datetime.now()
        total_loss = 0
        solved = []
        for truth_group, dp_group in zip(self.ground_truth.groupby(level='neighborhood'), dp_data.groupby(level='neighborhood')):
            truth_data = truth_group[1].values.astype('float')
            dp_data = dp_group[1].values.astype('float')
            # normalize
            # truth_data /= np.sum(truth_data)
            # dp_data /= np.sum(dp_data)
            print("max diff of columns", np.max(np.abs(np.sum(truth_data, axis=0) - np.sum(dp_data, axis=0))) )
            truth_data = truth_data.flatten()
            dp_data = dp_data.flatten()
            b = np.concatenate((alpha * (dp_data - Delta), -alpha * (dp_data + Delta), np.zeros(total_cells),
                                np.zeros(num_flow_variables)))

            x = Variable(num_flow_variables + 2 * total_cells)
            constraints = [A @ x >= b, F @ x == truth_data]
            objective = Minimize(c.T @ x)
            problem = Problem(objective, constraints)
            problem.solve(verbose=False, solver=ECOS)
            print(truth_group[0], 'problem state: ', problem.status, problem.value)
            x = np.array(x.value)
            print('solution x: ', x, np.max(x))
            abs_diff = np.sum(np.abs(truth_data - dp_data))
            print("abs diff v.s. AEMD:", abs_diff, problem.value)
            print("===============")
            total_loss += problem.value
            solved.append({'loss': problem.value, 'x': x.tolist()})
            self.save_json(solved, 'month_only_V2')
        print("total_time:", datetime.now() - start_t)
        print("total loss:", total_loss)
        return


    def _month_incident_dummy_V2(self, target_dp_data_label):
        print("calculate _month_incident_dummy")
        num_incident_type = 174
        num_incident_type_with_dummy = 175
        months = 12
        total_cells = num_incident_type_with_dummy * 12
        num_flow_variables = (10 * 3 + 2 * 2) * num_incident_type + months * num_incident_type
        Delta = self.Delta
        alpha = self.alpha
        dp_data = self.dp_data[target_dp_data_label]
        print("parameter: Delta/alpha", Delta, alpha)

        # design x = [flows_to_next_month] + [flows_to_previous_month] + [flow_within_month]
        to_pre_base_idx = 11 * num_incident_type
        within_month_base_idx = 22 * num_incident_type
        S = np.zeros((total_cells, num_flow_variables))
        N = np.zeros((total_cells, num_flow_variables))
        for i in range(total_cells):
            if (i + 1) % num_incident_type_with_dummy == 0:
                month = int((i + 1) / num_incident_type_with_dummy)
                related_idx_start = within_month_base_idx + (month - 1) * 2 * num_incident_type
                N[i, related_idx_start + num_incident_type: related_idx_start + 2 * num_incident_type] = 1
                N[i, related_idx_start: related_idx_start + num_incident_type] = -1
                S[i, related_idx_start: related_idx_start + num_incident_type] = 0.5
            elif i < num_incident_type_with_dummy:
                N[i, [i, within_month_base_idx + i]] = 1
                N[i, [to_pre_base_idx + i, within_month_base_idx + num_incident_type + i]] = -1
                S[i, to_pre_base_idx + i] = 1
                S[i, within_month_base_idx + num_incident_type + i] = 0.5
            elif i >= 11 * num_incident_type_with_dummy:
                incident_type = i % num_incident_type_with_dummy
                within_month_start_idx = within_month_base_idx + 22 * num_incident_type
                N[i, [to_pre_base_idx + 10 * num_incident_type + incident_type, within_month_start_idx + incident_type]] = 1
                N[i, [11 * num_incident_type + incident_type - num_incident_type, within_month_start_idx + num_incident_type + incident_type]] = -1
                S[i, 11 * num_incident_type + incident_type - num_incident_type] = 1
                S[i, within_month_start_idx + num_incident_type + incident_type] = 0.5
            else:
                m = int(i / num_incident_type_with_dummy)
                incident_type = i % num_incident_type_with_dummy
                within_month_start_idx = within_month_base_idx + m * num_incident_type
                N[i, [m * num_incident_type + incident_type, to_pre_base_idx + (m - 1) * num_incident_type + incident_type]] = 1
                N[i, within_month_start_idx + incident_type] = 1
                N[i, [m * num_incident_type + incident_type - num_incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = -1
                N[i, within_month_start_idx + num_incident_type +incident_type] = 1
                S[i, [m * num_incident_type + incident_type - num_incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = 1
                S[i, within_month_start_idx + num_incident_type +incident_type] = 0.5

        A = np.zeros((3 * total_cells + num_flow_variables, num_flow_variables + 2 * total_cells))
        A[:total_cells, :num_flow_variables] = -S
        A[:total_cells, num_flow_variables:num_flow_variables + total_cells] = alpha * np.eye(total_cells)
        A[:total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[total_cells:2 * total_cells, :num_flow_variables] = -S
        A[total_cells:2 * total_cells, num_flow_variables:num_flow_variables + total_cells] = -alpha * np.eye(
            total_cells)
        A[total_cells:2 * total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[2 * total_cells:3 * total_cells, :num_flow_variables] = -S
        A[2 * total_cells:3 * total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[3 * total_cells:, :num_flow_variables] = np.eye(num_flow_variables)

        F = np.zeros((total_cells, num_flow_variables + 2 * total_cells))
        F[:, :num_flow_variables] = N
        F[:, num_flow_variables:num_flow_variables + total_cells] = np.eye(total_cells)
        print("Initialize constraint matrix done")

        # distance = np.ones((num_flow_variables, 1))
        c = np.array([0] * (num_flow_variables + total_cells) + [1] * total_cells).reshape((-1, 1))
        print("Use 1 for all distances between neighbors")

        # todo: feed into solver
        start_t = datetime.now()
        total_loss = 0
        solved = []
        for truth_group, dp_group in zip(self.ground_truth.groupby(level='neighborhood'),
                                         dp_data.groupby(level='neighborhood')):
            truth_data = truth_group[1].values.astype('float')
            dp_data = dp_group[1].values.astype('float')
            # append dummy node
            truth_data = np.append(truth_data, np.zeros((months, 1)), 1)
            dp_data = np.append(dp_data, np.zeros((months, 1)), 1)
            print("max diff of columns", np.max(np.abs(np.sum(truth_data, axis=0) - np.sum(dp_data, axis=0))))
            truth_data = truth_data.flatten()
            dp_data = dp_data.flatten()
            b = np.concatenate((alpha * (dp_data - Delta), -alpha * (dp_data + Delta), np.zeros(total_cells),
                                np.zeros(num_flow_variables)))
            # b2 = np.concatenate((alpha * (truth_data - Delta), -alpha * (truth_data + Delta), np.zeros(total_cells),
            #                     np.zeros(num_flow_variables)))

            x = Variable(num_flow_variables + 2 * total_cells)
            constraints = [A @ x >= b, F @ x == truth_data]
            # constraints2 = [A @ x >= b2, F @ x == dp_data]
            objective = Minimize(c.T @ x)
            # objective2 = Minimize(c.T @ x)
            problem = Problem(objective, constraints)
            # problem2 = Problem(objective2, constraints2)
            problem.solve(verbose=False, solver=ECOS)
            # problem2.solve(verbose=False, solver=ECOS)
            print(truth_group[0], 'problem state: ', problem.status, problem.value)
            # print(truth_group[0], 'problem2 state: ', problem.status, problem2.value)
            x = np.array(x.value)
            print('solution x: ', x, np.max(x))
            total_loss += problem.value
            abs_diff = np.sum(np.abs(truth_data - dp_data))
            print("abs diff v.s. AEMD:", abs_diff, problem.value)
            print("===============")
            solved.append({'loss': problem.value, 'x': x.tolist()})
            self.save_json(solved, 'month_incident_dummy_V2')
        print("total_time:", datetime.now() - start_t)
        print("total loss:", total_loss)

        return


    def _month_only_flow(self, target_dp_data_label):
        num_incident_type = 174
        total_cells = num_incident_type * 12
        num_flow_variables = total_cells + 2    # source and sink
        Delta = self.Delta
        alpha = self.alpha
        dp_data = self.dp_data[target_dp_data_label]
        print("parameter: Delta/alpha", Delta, alpha)

        def index(incident_type, month):
            return month * num_incident_type + incident_type

        print("calculate _month_only_flow")

        # todo: feed into solver
        start_t = datetime.now()
        total_loss = 0
        solved = []
        for truth_group, dp_group in zip(self.ground_truth.groupby(level='neighborhood'), dp_data.groupby(level='neighborhood')):
            problem_value, capacity_scale, cost_scale, inf = 0, 1, 1, 500000
            edges, node_demands = [], [0] * num_flow_variables
            source, sink = total_cells, total_cells + 1

            def add_edge(u, v, demand, capacity, cost):
                nonlocal capacity_scale, cost_scale, problem_value, edges, node_demands
                real_capacity = capacity - demand
                assert(real_capacity >= 0)
                capacity_scale = math.lcm(capacity_scale, real_capacity.as_integer_ratio()[1], demand.as_integer_ratio()[1])
                cost_scale = math.lcm(cost_scale, cost.as_integer_ratio()[1])
                # print(u, v, num_flow_variables, len(node_demands))
                node_demands[u] += demand
                node_demands[v] -= demand
                if real_capacity > 0:
                    edges.append((u, v, real_capacity, cost))
                problem_value += demand * cost
            
            # add indendent edges
            add_edge(sink, source, 0, inf, 0)
            for i in range(total_cells):
                month = i // num_incident_type
                if month < 11:
                    add_edge(i, i + num_incident_type, 0, inf, 1)
                if month > 0:
                    add_edge(i, i - num_incident_type, 0, inf, 1)
                add_edge(source, i, 0, inf, alpha)
                add_edge(i, sink, 0, inf, alpha)

            # add data
            truth_data = truth_group[1].values.astype('float')
            dp_data = dp_group[1].values.astype('float')
            print("max diff of columns", np.max(np.abs(np.sum(truth_data, axis=0) - np.sum(dp_data, axis=0))) )
            truth_data = truth_data.flatten()
            dp_data = dp_data.flatten()
            for i in range(total_cells):
                add_edge(source, i, truth_data[i], truth_data[i], 0)
                add_edge(i, sink, dp_data[i] - Delta, dp_data[i] + Delta, 0)

            # problem_value = round(problem_value * cost_scale * capacity_scale)

            # solve
            min_cost_flow = pywrapgraph.SimpleMinCostFlow()
            for (u, v, capacity, cost) in edges:
                min_cost_flow.AddArcWithCapacityAndUnitCost(u, v, round(capacity * capacity_scale), round(cost * cost_scale))

            for i in range(num_flow_variables):
                min_cost_flow.SetNodeSupply(i, -round(node_demands[i] * capacity_scale))
                # print(i, round(node_demands[i] * capacity_scale))

            assert(min_cost_flow.Solve() == min_cost_flow.OPTIMAL)
            problem_value += min_cost_flow.OptimalCost()
            problem_value = problem_value / capacity_scale / cost_scale
            # print(truth_data)
            # print(dp_data)
            # b = np.concatenate((alpha * (dp_data - Delta), -alpha * (dp_data + Delta), np.zeros(total_cells),
            #                     np.zeros(num_flow_variables)))

            # x = Variable(num_flow_variables + 2 * total_cells)
            # constraints = [A @ x >= b, F @ x == truth_data]
            # objective = Minimize(c.T @ x)
            # problem = Problem(objective, constraints)
            # problem.solve(verbose=False, solver=ECOS)
            # print(truth_group[0], 'problem state: ', problem.status, problem.value)
            # x = np.array(x.value)
            # print('solution x: ', x, np.max(x))
            abs_diff = np.sum(np.abs(truth_data - dp_data))
            print("abs diff v.s. AEMD:", abs_diff, problem_value)
            print("===============")
            total_loss += problem_value
            solved.append({'loss': problem_value})
            self.save_json(solved, 'month_only_flow')
        print("total_time:", datetime.now() - start_t)
        print("total loss:", total_loss)
        return

    def _simplest(self, target_dp_data_label):
        print("calculate _exact_emd")
        # todo: initial distance matrix
        total_variables = 6
        num_cell = 2
        alpha = 1.5
        Delta = 1
        truth_data = np.array([4, 1])
        dp_data = np.array([2, 3])

        distance = np.array([1, 1])
        neighbor_matirx = np.zeros((num_cell, num_cell))
        neighbor_matirx[0, 0] = 1
        neighbor_matirx[0, 1] = -1
        neighbor_matirx[1, 0] = -1
        neighbor_matirx[1, 1] = 1

        x = Variable(2)
        constraints = [neighbor_matirx @ x - (truth_data - dp_data) >= -Delta,
                       neighbor_matirx @ x - (truth_data - dp_data) <= Delta,
                       x >= 0]
        objective = Minimize(distance @ x)
        problem = Problem(objective, constraints)
        problem.solve(verbose=True)
        print('problem state: ', problem.status)
        print('solution: ', np.array(x.value))
        exit()
