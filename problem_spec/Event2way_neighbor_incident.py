'''
    This part follows the algorithm designed by Trung Dang (@kuroni)
'''

from .Base import Base
import numpy as np
from datetime import datetime
from ortools.graph import pywrapgraph
from fractions import Fraction


class Event2way_neighbor_incident(Base):
    def __init__(self, Delta, alpha=1):
        super().__init__()
        self.Delta = Delta
        self.alpha = alpha

    def _to_2way_marginal(self, P, Q):
        P = P.groupby('neighborhood').sum()
        Q = Q.groupby('neighborhood').sum()
        return P, Q

    def compute_AEMC(self, P, Q):
        P, Q = self._to_2way_marginal(P, Q)
        num_incident_type = 174
        num_neighbor = 278
        total_cells = num_incident_type * num_neighbor
        num_flow_variables = total_cells * 2 + 2 + num_neighbor + num_incident_type # source, sink, and 12 dummy nodes
        Delta = self.Delta
        alpha = self.alpha
        print("parameter: Delta/alpha", Delta, alpha)

        print("calculate _neighbour_incident_dummy_flow_V2")

        start_t = datetime.now()

        problem_value, capacity_scale, cost_scale, inf = 0, 1, 1, 50000000
        edges, node_demands = [], [0] * num_flow_variables
        source, sink,  = 2 * total_cells, 2 * total_cells + 1
        neighborhood_dummy, incident_dummy = 2 * total_cells + 2, total_cells * 2 + 2 + num_incident_type

        def denominator(number):
            return Fraction.from_float(number).limit_denominator().denominator

        def add_edge(u, v, demand, capacity, cost):
            nonlocal capacity_scale, cost_scale, problem_value, edges, node_demands
            real_capacity = capacity - demand
            assert (real_capacity >= 0)
            # capacity_scale = math.lcm(capacity_scale, denominator(real_capacity), denominator(demand))
            # cost_scale = math.lcm(cost_scale, denominator(cost))
            capacity_scale = np.lcm.reduce([capacity_scale, denominator(real_capacity), denominator(demand)])
            cost_scale = np.lcm.reduce([cost_scale, denominator(cost)])
            # print(u, v, num_flow_variables, len(node_demands))
            node_demands[u] += demand
            node_demands[v] -= demand
            if real_capacity > 0:
                edges.append((u, v, real_capacity, cost))
            problem_value += demand * cost

        # add indendent edges
        add_edge(sink, source, 0, inf, 0)
        for i in range(total_cells):
            neighborhood = i // num_incident_type
            incident_type = i % num_incident_type
            add_edge(i, i + total_cells, -inf, inf, 0)
            add_edge(source, i + total_cells, 0, inf, alpha)
            add_edge(i + total_cells, sink, 0, inf, alpha)
            add_edge(i, incident_dummy + neighborhood, 0, inf, 0.5 * 0.5)
            add_edge(incident_dummy + neighborhood, i, 0, inf, 0.5 * 0.5)
            add_edge(i, neighborhood_dummy + incident_type, 0, inf, 0.5 * 0.5)
            add_edge(neighborhood_dummy + incident_type, i, 0, inf, 0.5 * 0.5)

        # add data
        P = P.values.astype('float')
        Q = Q.values.astype('float')
        # print("max diff of columns", np.max(np.abs(np.sum(P, axis=0) - np.sum(Q, axis=0))))
        P = P.flatten()
        Q = Q.flatten()
        for i in range(total_cells):
            add_edge(source, i, P[i], P[i], 0)
            add_edge(i + total_cells, sink, Q[i] - Delta, Q[i] + Delta, 0)

        problem_value = round(problem_value * cost_scale * capacity_scale)

        # solve
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for (u, v, capacity, cost) in edges:
            min_cost_flow.AddArcWithCapacityAndUnitCost(u, v, round(capacity * capacity_scale),
                                                        round(cost * cost_scale))

        for i in range(num_flow_variables):
            min_cost_flow.SetNodeSupply(i, -round(node_demands[i] * capacity_scale))
            # print(i, round(node_demands[i] * capacity_scale))

        assert (min_cost_flow.Solve() == min_cost_flow.OPTIMAL)
        problem_value += min_cost_flow.OptimalCost()
        problem_value = problem_value / capacity_scale / cost_scale

        abs_diff = np.sum(np.abs(P - Q))
        print("abs diff v.s. AEMD:", abs_diff, problem_value)
        print("total_time:", datetime.now() - start_t)
        print("===============")
        return problem_value
