'''
    This algorithm is desgined and implemented by Trung Dang (@kuroni)
'''
from .Base import Base
import numpy as np
from datetime import datetime
from ortools.graph import pywrapgraph
from fractions import Fraction


class Event3way(Base):
    def __init__(self, Delta, alpha=1):
        super().__init__()
        self.Delta = Delta
        self.alpha = alpha


    def compute_AEMC(self, ground_truth, dp_data):
        num_incident_type = 174
        num_neighborhood_type = 278
        total_cells = num_incident_type * num_neighborhood_type * 12
        num_flow_variables = total_cells + 2 + 12 * num_neighborhood_type + 12 * num_incident_type # source, sink, dummy nodes
        Delta = self.Delta
        alpha = self.alpha
        print("parameter: Delta/alpha", Delta, alpha)

        print("calculate _three_way_margin_flow")

        start_t = datetime.now()
        # total_loss = 0
        # solved = []

        problem_value, capacity_scale, cost_scale, inf = 0, 1, 1, 50000000
        edges, node_demands = [], [0] * num_flow_variables
        source, sink = total_cells, total_cells + 1
        incident_dummy, neighborhood_dummy = total_cells + 2, total_cells + 2 + 12 * num_neighborhood_type

        def denominator(number):
            # return number.as_integer_ratio()[1]
            return Fraction.from_float(number).limit_denominator().denominator

        def add_edge(u, v, demand, capacity, cost):
            nonlocal capacity_scale, cost_scale, problem_value, edges, node_demands
            real_capacity = capacity - demand
            assert(real_capacity >= 0)
            # capacity_scale = math.lcm(capacity_scale, denominator(real_capacity), denominator(demand))
            # cost_scale = math.lcm(cost_scale, denominator(cost))
            capacity_scale = np.lcm.reduce([capacity_scale, denominator(real_capacity), denominator(demand)])
            cost_scale = np.lcm.reduce([cost_scale, denominator(cost)])
            node_demands[u] += demand
            node_demands[v] -= demand
            if real_capacity > 0:
                edges.append((u, v, real_capacity, cost))
            problem_value += demand * cost
        
        # add indendent edges
        add_edge(sink, source, 0, inf, 0)
        for i in range(total_cells):
            neighborhood = i // (num_incident_type * 12)
            month = (i - neighborhood * num_incident_type * 12) // num_incident_type
            incident_type = i - neighborhood * num_incident_type * 12 - month * num_incident_type
            if month < 11:
                add_edge(i, i + num_incident_type, 0, inf, 1 * 0.33)
            if month > 0:
                add_edge(i, i - num_incident_type, 0, inf, 1 * 0.33)
            add_edge(source, i, 0, inf, alpha)
            add_edge(i, sink, 0, inf, alpha)
            add_edge(i, incident_dummy + neighborhood * 12 + month, 0, inf, 0.5 * 0.33)
            add_edge(incident_dummy + neighborhood * 12 + month, i, 0, inf, 0.5 * 0.33)
            add_edge(i, neighborhood_dummy + incident_type * 12 + month, 0, inf, 0.5 * 0.33)
            add_edge(neighborhood_dummy + incident_type * 12 + month, i, 0, inf, 0.5 * 0.33)

        # add data
        truth_data = ground_truth.values.astype('float')
        dp_data = dp_data.values.astype('float')
        print("max diff of columns", np.max(np.abs(np.sum(truth_data, axis=0) - np.sum(dp_data, axis=0))) )
        truth_data = truth_data.flatten()
        dp_data = dp_data.flatten()
        for i in range(total_cells):
            add_edge(source, i, truth_data[i], truth_data[i], 0)
            add_edge(i, sink, dp_data[i] - Delta, dp_data[i] + Delta, 0)

        problem_value = round(problem_value * cost_scale * capacity_scale)

        # solve
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for (u, v, capacity, cost) in edges:
            min_cost_flow.AddArcWithCapacityAndUnitCost(u, v, round(capacity * capacity_scale), round(cost * cost_scale))

        for i in range(num_flow_variables):
            min_cost_flow.SetNodeSupply(i, -round(node_demands[i] * capacity_scale))

        assert(min_cost_flow.Solve() == min_cost_flow.OPTIMAL)
        problem_value += min_cost_flow.OptimalCost()
        problem_value = problem_value / capacity_scale / cost_scale
        abs_diff = np.sum(np.abs(truth_data - dp_data))
        print("abs diff v.s. AEMD:", abs_diff, problem_value)
        # total_loss += problem_value
        # solved.append({'loss': problem_value})
        # self.save_json(solved, '_three_way_margin_flow')
        print("total_time:", datetime.now() - start_t)
        # print("total loss:", total_loss)
        print("===============")
        return problem_value
