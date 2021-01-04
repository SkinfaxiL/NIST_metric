from .Base import Base
import numpy as np
from datetime import datetime
from cvxpy import *

class Event2wayLP_month_neighbor(Base):
    def __init__(self, Delta, alpha=1):
        super().__init__()
        self.Delta = Delta
        self.alpha = alpha

    def _to_2way_marginal(self, ground_truth, dp_data):
        ground_truth = ground_truth.groupby('month').sum()
        dp_data = dp_data.groupby('month').sum()
        return ground_truth, dp_data

    def compute_AEMC(self, ground_truth, dp_data):
        ground_truth, dp_data = self._to_2way_marginal(ground_truth, dp_data)
        num_incident_type = 174
        num_incident_type_with_dummy = 175
        months = 12
        total_cells = num_incident_type_with_dummy * 12
        num_flow_variables = (10 * 3 + 2 * 2) * num_incident_type + months * num_incident_type
        Delta = self.Delta
        alpha = self.alpha
        print("parameter: Delta/alpha", Delta, alpha)

        # design x = [flows_to_next_month] + [flows_to_previous_month] + [flow_within_month]
        to_pre_base_idx = 11 * num_incident_type
        within_month_base_idx = 22 * num_incident_type
        S = np.zeros((total_cells, num_flow_variables))
        N = np.zeros((total_cells, num_flow_variables))
        for i in range(total_cells):
            if (i + 1) % num_incident_type_with_dummy == 0:
                month = int(i / num_incident_type_with_dummy)
                related_idx_start = within_month_base_idx + month * 2 * num_incident_type
                # flow out
                N[i, related_idx_start + num_incident_type: related_idx_start + 2 * num_incident_type] = 1
                # flow in
                N[i, related_idx_start: related_idx_start + num_incident_type] = -1
                S[i, related_idx_start: related_idx_start + num_incident_type] = 0.5 * 0.5
            elif i < num_incident_type:
                # flow out
                N[i, [i, within_month_base_idx + i]] = 1
                # flow in
                N[i, [to_pre_base_idx + i, within_month_base_idx + num_incident_type + i]] = -1
                S[i, to_pre_base_idx + i] = 1 * 0.5
                S[i, within_month_base_idx + num_incident_type + i] = 0.5 * 0.5
            elif i >= 11 * num_incident_type_with_dummy:
                incident_type = i % num_incident_type_with_dummy
                within_month_start_idx = within_month_base_idx + 22 * num_incident_type
                N[i, [to_pre_base_idx + 10 * num_incident_type + incident_type, within_month_start_idx + incident_type]] = 1
                N[i, [10 * num_incident_type + incident_type, within_month_start_idx + num_incident_type + incident_type]] = -1
                S[i, 11 * num_incident_type + incident_type - num_incident_type] = 1 * 0.5
                S[i, within_month_start_idx + num_incident_type + incident_type] = 0.5 * 0.5
            else:
                m = int(i / num_incident_type_with_dummy)
                incident_type = i % num_incident_type_with_dummy
                within_month_start_idx = within_month_base_idx + 2 * m * num_incident_type
                N[i, [m * num_incident_type + incident_type, to_pre_base_idx + (m - 1) * num_incident_type + incident_type]] = 1
                N[i, within_month_start_idx + incident_type] = 1
                N[i, [(m - 1) * num_incident_type + incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = -1
                N[i, within_month_start_idx + num_incident_type + incident_type] = 1
                S[i, [(m - 1) * num_incident_type + incident_type, to_pre_base_idx + m * num_incident_type + incident_type]] = 1  * 0.5
                S[i, within_month_start_idx + num_incident_type + incident_type] = 0.5 * 0.5

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

        F = np.zeros((total_cells + months, num_flow_variables + 2 * total_cells))
        F[:total_cells, :num_flow_variables] = N
        F[:total_cells, num_flow_variables:num_flow_variables + total_cells] = np.eye(total_cells)
        for m in range(months):
            # print(num_incident_type_with_dummy * m + num_incident_type)
            F[total_cells + m, num_flow_variables + num_incident_type_with_dummy * m + num_incident_type] = 1
        print("Initialize constraint matrix done")

        # distance = np.ones((num_flow_variables, 1))
        c = np.array([0] * (num_flow_variables + total_cells) + [1] * total_cells).reshape((-1, 1))
        print("Use 1 for all distances between neighbors")

        # feed into solver
        start_t = datetime.now()
        total_loss = 0
        solved = []
        truth_data = ground_truth.values.astype('float')
        dp_data = dp_data.values.astype('float')
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
        truth_data_appended = np.append(truth_data, np.zeros(months), 0)
        constraints = [A @ x >= b, F @ x == truth_data_appended]
        # constraints2 = [A @ x >= b2, F @ x == dp_data]
        objective = Minimize(c.T @ x)
        # objective2 = Minimize(c.T @ x)
        problem = Problem(objective, constraints)
        # problem2 = Problem(objective2, constraints2)
        problem.solve(verbose=False, solver=ECOS)
        # problem2.solve(verbose=False, solver=ECOS)
        print('problem state: ', problem.status, problem.value)
        # print(truth_group[0], 'problem2 state: ', problem.status, problem2.value)
        x = np.array(x.value)
        print('solution x: ', x, np.max(x))
        print("===============")
        print("total_time:", datetime.now() - start_t)

        return problem.value