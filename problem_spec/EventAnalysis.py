from .Base import Base
import pandas as pd
import os
import numpy as np
from cvxopt import matrix, solvers


class EventAnalysis(Base):
    def __init__(self):
        super().__init__()
        self.file_dir = './data/Event'
        self.ground_truth_file = 'ground_truth.csv'
        self.ground_truth = None
        self.dp_data_files = {"mid": 'privatized-mediocre-quality.csv',
                              "poor": 'privatized-very-poor-quality.csv'}
        self.ground_truth = pd.read_csv(os.path.join(self.file_dir, self.ground_truth_file),
                                        index_col=["neighborhood", "year", "month"])
        self.dp_data = {}
        read_cols = ["neighborhood", "year", "month"] + [str(i) for i in range(174)]
        for label, filename in self.dp_data_files.items():
            self.dp_data[label] = pd.read_csv(os.path.join(self.file_dir, filename),
                                              index_col=["neighborhood", "year", "month"],
                                              usecols=read_cols)

        self.compare_strategy = {"month_incident_fixed": self._month_incident_fixed,
                                 "month_only": self._month_only}

    def calculate_AEMD(self, target_dp_data_label, compare_strategy):
        assert target_dp_data_label in self.dp_data
        assert compare_strategy in self.compare_strategy
        return self.compare_strategy[compare_strategy](target_dp_data_label)

    def _month_incident_fixed(self, target_dp_data_label):
        print("calculate _month_incident_fixed")
        # todo: initial distance matrix
        num_flow = 3
        num_incident_type = 174
        months = 12
        alpha = 1.5
        Delta = 1
        dp_data = self.dp_data[target_dp_data_label]

        def get_flow_in(i):
            if i >= num_incident_type and i < num_incident_type * (months - 1):
                return [i, i], [(i - num_incident_type) * num_flow + 2, (i + num_incident_type) * num_flow ]
            elif i < num_incident_type:
                return [i], [(i + num_incident_type) * num_flow ]
            else:
                return [i], [(i - num_incident_type) * num_flow + 2]

        def get_flow_out(i):
            if i >= num_incident_type and i < num_incident_type * (months - 1):
                return [i, i], [i * num_flow, i * num_flow + 2]
            elif i < num_incident_type:
                return [i], [i * num_flow + 2]
            else:
                return [i], [i * num_flow]

        distance_matrix = np.zeros([num_incident_type * months, num_incident_type * months * num_flow])
        neighbor_matrix = np.zeros([num_incident_type * months, num_incident_type * months * num_flow])



        v_mapping = np.array([i * num_flow for i in range(num_incident_type * months)])
        flow_in_mapping = [[], []]
        flow_out_mapping = [[], []]
        for v in range(num_incident_type * months):
            vs, flow_ins = get_flow_in(v)
            flow_in_mapping[0] += vs
            flow_in_mapping[1] += flow_ins
            vs, flow_outs = get_flow_out(v)
            flow_out_mapping[0] += vs
            flow_out_mapping[1] += flow_outs

        neighbor_matrix[[i for i in range(num_incident_type * months)], v_mapping + 1] = 1
        print(v_mapping)
        neighbor_matrix[flow_out_mapping[0], flow_out_mapping[1]] = -1
        neighbor_matrix[flow_in_mapping[0], flow_in_mapping[1]] = 1

        # distance_matrix[flow_out_mapping[0], flow_out_mapping[1]] = 1
        distance_matrix[flow_in_mapping[0], flow_in_mapping[1]] = 1

        diagnol_m = np.eye(num_incident_type * months)

        A = np.zeros(((4 + num_flow)* num_incident_type * months, (num_flow + 1) * num_incident_type * months))
        A[: num_incident_type * months, :num_incident_type * months * num_flow] = - distance_matrix + alpha * neighbor_matrix
        A[: num_incident_type * months, num_incident_type * months * num_flow:] = diagnol_m
        A[num_incident_type * months:2*num_incident_type * months, :num_incident_type * months * num_flow] = - distance_matrix - alpha * neighbor_matrix
        A[num_incident_type * months:2*num_incident_type * months, num_incident_type * months * num_flow:] = diagnol_m
        A[2 * num_incident_type * months:3 * num_incident_type * months, :num_incident_type * months * num_flow] = neighbor_matrix
        A[3 * num_incident_type * months:4 * num_incident_type * months, :num_incident_type * months * num_flow] = - neighbor_matrix
        A[4 * num_incident_type * months:, :num_incident_type * months * num_flow] = np.eye(num_incident_type * months * num_flow)

        print(np.unique(distance_matrix.sum(axis=1), return_counts=True), np.unique(np.abs(neighbor_matrix).sum(axis=1), return_counts=True))
        print(np.where(neighbor_matrix[0] == 1), np.where(neighbor_matrix[0] == -1))
        print(np.where(neighbor_matrix[348] == 1), np.where(neighbor_matrix[348] == -1))
        print(np.where(distance_matrix[0] == 1))
        print(np.where(distance_matrix[348] == 1))
        print(np.where(A[0] != 0), A[0, 1], A[0, 2], A[0, 522], A[0, 6264])

        c = np.ones((num_flow + 1) * num_incident_type * months)
        c[:num_flow * num_incident_type * months] = 0

        # todo: feed into solver
        for truth_group, dp_group in zip(self.ground_truth.groupby(level='neighborhood'), dp_data.groupby(level='neighborhood')):
            truth_data = truth_group[1].values.flatten()
            dp_data = dp_group[1].values.flatten()
            print(truth_data.shape, dp_data.shape)
            A = matrix(A)
            c = matrix(c)
            b = np.concatenate((alpha * dp_data, -alpha * dp_data, truth_data - Delta, - truth_data - Delta,
                                np.zeros(num_incident_type * months * num_flow))).reshape((-1, 1))
            b = b.astype('float')
            b = matrix(b)
            print("b size", b.size)
            sol = solvers.lp(c, G=A, h=b)
            print(sol['primal objective'], sol['status'] )
            exit()

        return

    def _month_only(self, target_dp_data_label):
        print("calculate _month_only")
        # todo: initial distance matrix
        num_flow = 3
        months = 12
        alpha = 0.1
        Delta = 1
        dp_data = self.dp_data[target_dp_data_label]

        distance_matrix = np.zeros([months, months * num_flow])
        neighbor_matrix = np.zeros([months, months * num_flow])
        neighbor_matrix_inverse = np.zeros([months, months * num_flow])

        def get_flow_in(i):
            if i >= 1 and i < (months - 1):
                return [i, i], [(i - 1) * num_flow + 2, (i + 1) * num_flow]
            elif i == 0:
                return [i], [num_flow]
            else:
                return [i], [(i - 1) * num_flow + 2]

        def get_flow_out(i):
            if i >= 1 and i < months - 1:
                return [i, i], [i * num_flow, i * num_flow + 2]
            elif i == 0:
                return [i], [2]
            else:
                return [i], [i * num_flow]

        v_mapping = np.array([i * num_flow for i in range(months)])
        flow_in_mapping = [[], []]
        flow_out_mapping = [[], []]
        for v in range(months):
            vs, flow_ins = get_flow_in(v)
            flow_in_mapping[0] += vs
            flow_in_mapping[1] += flow_ins
            vs, flow_outs = get_flow_out(v)
            flow_out_mapping[0] += vs
            flow_out_mapping[1] += flow_outs
            print(flow_ins, flow_outs)

        neighbor_matrix[[i for i in range(months)], v_mapping + 1] = 1
        neighbor_matrix[flow_out_mapping[0], flow_out_mapping[1]] = -1
        neighbor_matrix[flow_in_mapping[0], flow_in_mapping[1]] = 1

        neighbor_matrix_inverse[[i for i in range(months)], v_mapping + 1] = 1
        neighbor_matrix_inverse[flow_out_mapping[0], flow_out_mapping[1]] = 1
        neighbor_matrix_inverse[flow_in_mapping[0], flow_in_mapping[1]] = -1


        distance_matrix[flow_in_mapping[0], flow_in_mapping[1]] = 1
        # distance_matrix[0, 0] = 99999
        # distance_matrix[-1, -1] = 99999
        print(neighbor_matrix)
        print("===")
        print(distance_matrix)

        diagnol_m = np.eye(months)

        A = np.zeros(((4 + num_flow) * months, (num_flow + 1) * months))
        A[: months, : months * num_flow] = - distance_matrix + alpha * neighbor_matrix
        A[: months,  months * num_flow:] = diagnol_m
        A[months:2 * months, : months * num_flow] = - distance_matrix - alpha * neighbor_matrix
        A[months:2 * months,  months * num_flow:] = diagnol_m
        A[2 * months:3 * months, : months * num_flow] = neighbor_matrix_inverse
        A[3 * months:4 * months, : months * num_flow] = - neighbor_matrix_inverse
        A[4 * months:, : months * num_flow] = np.eye(months * num_flow)
        print(A)

        print(np.unique(distance_matrix.sum(axis=1), return_counts=True), np.unique(np.abs(neighbor_matrix).sum(axis=1), return_counts=True))

        c = np.ones((num_flow + 1) * months)
        c[:num_flow * months] = 0

        # todo: feed into solver
        for truth_group, dp_group in zip(self.ground_truth.groupby(level='neighborhood'), dp_data.groupby(level='neighborhood')):
            truth_data = truth_group[1].values[:,0].flatten()
            dp_data = dp_group[1].values[:,0].flatten()
            print(truth_data, dp_data)
            A = matrix(A)
            c = matrix(c)
            b = np.concatenate((alpha * dp_data, -alpha * dp_data, truth_data - Delta, - truth_data - Delta,
                                np.zeros(months * num_flow))).reshape((-1, 1))
            b = b.astype('float')
            b = matrix(b)
            print("b size", b.size)
            sol = solvers.lp(c, G=A, h=b)
            print(sol['primal objective'], sol['status'], sol['x'])
            exit()

        return