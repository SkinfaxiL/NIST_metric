from .Base import Base
import numpy as np
from datetime import datetime
from cvxpy import *

'''
    This AEMC only allows movement between "month", not allow movement between "incident types"
'''
class Event2wayLP_month_neighbor2(Base):
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
        total_cells = num_incident_type * 12
        num_flow_variables = (10 * 2 + 2) * num_incident_type
        Delta = self.Delta
        alpha = self.alpha
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
        A[total_cells:2 * total_cells, num_flow_variables:num_flow_variables + total_cells] = -alpha * np.eye(
            total_cells)
        A[total_cells:2 * total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[2 * total_cells:3 * total_cells, :num_flow_variables] = -S
        A[2 * total_cells:3 * total_cells, num_flow_variables + total_cells:] = np.eye(total_cells)
        A[3 * total_cells:, :num_flow_variables] = np.eye(num_flow_variables)
        print("Initialize constraint matrix done")

        F = np.zeros((total_cells, num_flow_variables + 2 * total_cells))
        F[:, :num_flow_variables] = N
        F[:, num_flow_variables:num_flow_variables + total_cells] = np.eye(total_cells)
        # distance = np.ones((num_flow_variables, 1))
        c = np.array([0] * (num_flow_variables + total_cells) + [1] * total_cells).reshape((-1, 1))

        # feed into solver
        start_t = datetime.now()
        truth_data = ground_truth.values.astype('float')
        dp_data = dp_data.values.astype('float')
        print("max diff of columns", np.max(np.abs(np.sum(truth_data, axis=0) - np.sum(dp_data, axis=0))))
        truth_data = truth_data.flatten()
        dp_data = dp_data.flatten()
        b = np.concatenate((alpha * (dp_data - Delta), -alpha * (dp_data + Delta), np.zeros(total_cells),
                            np.zeros(num_flow_variables)))

        x = Variable(num_flow_variables + 2 * total_cells)
        constraints = [A @ x >= b, F @ x == truth_data]
        objective = Minimize(c.T @ x)
        problem = Problem(objective, constraints)
        problem.solve(verbose=False, solver=ECOS)
        print('problem state: ', problem.status, problem.value)
        print("===============")
        print("total_time:", datetime.now() - start_t)
        return problem.value
