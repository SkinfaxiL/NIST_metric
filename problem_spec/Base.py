import json
from datetime import datetime
import os

class Base:
    def __init__(self):
        self.start_time = datetime.now().strftime('%m-%d-%H-%M-%S-%f')
        self.compare_strategy = ''
        pass

    def load_csv(self):
        pass

    def solve(self):
        pass

    def generate_constraints(self):
        pass

    def save_json(self, result, file_name, results_dir='./result'):
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        json_file_name = os.path.join(results_dir, file_name+ '-' + self.start_time + '.json')
        with open(json_file_name, 'w') as f:
            json.dump(result, f, indent=2)
