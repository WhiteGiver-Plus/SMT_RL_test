from z3 import *
import numpy as np
import gym
from typing import List, Tuple, Dict
import time

class TaxiScenarioGenerator:
    def __init__(self):
        # Define the grid size
        self.GRID_SIZE = 5
        # Define the fixed locations (R, G, B, Y)
        self.LOCATIONS = {
            'R': (0, 0),
            'G': (0, 4),
            'B': (4, 0),
            'Y': (4, 3)
        }
        
    def create_scenario_constraints(self):
        # Create Z3 solver
        solver = Solver()
        
        # Variables for taxi position
        taxi_x = Int('taxi_x')
        taxi_y = Int('taxi_y')
        
        # Variables for passenger location (0-4 for locations, 5 for in taxi)
        passenger_loc = Int('passenger_loc')
        
        # Variable for destination (0-3 for locations)
        destination = Int('destination')
        
        # Basic constraints for taxi position
        solver.add(taxi_x >= 0, taxi_x < self.GRID_SIZE)
        solver.add(taxi_y >= 0, taxi_y < self.GRID_SIZE)
        
        # Constraints for passenger location and destination
        solver.add(passenger_loc >= 0, passenger_loc <= 5)  # 5 means in taxi
        solver.add(destination >= 0, destination <= 3)
        
        return solver, (taxi_x, taxi_y, passenger_loc, destination)
    
    def add_normal_scenario_constraints(self, solver, vars):
        taxi_x, taxi_y, passenger_loc, destination = vars
        # In normal scenarios, taxi is not too far from passenger
        # and destination is different from passenger location
        solver.add(Or(
            passenger_loc == 5,  # passenger in taxi
            And(Abs(taxi_x - self.LOCATIONS['R'][0]) + Abs(taxi_y - self.LOCATIONS['R'][1]) <= 3),
            And(Abs(taxi_x - self.LOCATIONS['G'][0]) + Abs(taxi_y - self.LOCATIONS['G'][1]) <= 3),
            And(Abs(taxi_x - self.LOCATIONS['B'][0]) + Abs(taxi_y - self.LOCATIONS['B'][1]) <= 3),
            And(Abs(taxi_x - self.LOCATIONS['Y'][0]) + Abs(taxi_y - self.LOCATIONS['Y'][1]) <= 3)
        ))
        
    def add_extreme_scenario_constraints(self, solver, vars):
        taxi_x, taxi_y, passenger_loc, destination = vars
        # In extreme scenarios, taxi might be far from both passenger and destination
        # or passenger and destination might be at opposite corners
        solver.push()
        solver.add(Or(
            And(taxi_x == 0, taxi_y == 0, passenger_loc == 3),  # Taxi at one corner, passenger at opposite
            And(taxi_x == 4, taxi_y == 4, passenger_loc == 0),
            And(passenger_loc == 0, destination == 3),  # Passenger and destination at opposite corners
            And(passenger_loc == 1, destination == 2)
        ))
        
    def generate_scenarios(self, num_normal: int = 5, num_extreme: int = 5) -> List[Dict]:
        scenarios = []
        
        # Generate normal scenarios
        for _ in range(num_normal):
            solver, vars = self.create_scenario_constraints()
            self.add_normal_scenario_constraints(solver, vars)
            
            if solver.check() == sat:
                model = solver.model()
                scenario = self._extract_scenario(model, vars)
                scenarios.append({"type": "normal", "scenario": scenario})
        
        # Generate extreme scenarios
        for _ in range(num_extreme):
            solver, vars = self.create_scenario_constraints()
            self.add_extreme_scenario_constraints(solver, vars)
            
            if solver.check() == sat:
                model = solver.model()
                scenario = self._extract_scenario(model, vars)
                scenarios.append({"type": "extreme", "scenario": scenario})
        
        return scenarios
    
    def _extract_scenario(self, model, vars):
        taxi_x, taxi_y, passenger_loc, destination = vars
        return {
            "taxi_position": (model[taxi_x].as_long(), model[taxi_y].as_long()),
            "passenger_location": model[passenger_loc].as_long(),
            "destination": model[destination].as_long()
        }
    
    def visualize_scenario(self, scenario):
        try:
            # Create environment with explicit render mode
            env = gym.make('Taxi-v3', render_mode='human')
            # Convert our scenario format to env's internal state
            state = self._convert_to_env_state(scenario)
            env.reset()
            env.env.s = state
            # Allow time for rendering
            time.sleep(0.5)
            return env
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Falling back to text-based visualization...")
            self._print_text_visualization(scenario)
            return None
    
    def _print_text_visualization(self, scenario):
        """Fallback text-based visualization"""
        grid = [[' ' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # Place taxi
        tx, ty = scenario["taxi_position"]
        grid[tx][ty] = 'T'
        
        # Place locations
        for name, (x, y) in self.LOCATIONS.items():
            grid[x][y] = name
        
        # Print the grid
        print("\n=== Scenario ===")
        print("Taxi at:", scenario["taxi_position"])
        print("Passenger at:", scenario["passenger_location"])
        print("Destination:", scenario["destination"])
        print("\nGrid visualization:")
        for row in grid:
            print('|' + '|'.join(row) + '|')
        print("===============\n")
    
    def _convert_to_env_state(self, scenario):
        # Taxi-v3 state is encoded as: taxi_row * 25 + taxi_col * 5 + passenger_location * 5 + destination
        taxi_x, taxi_y = scenario["taxi_position"]
        passenger_loc = scenario["passenger_location"]
        destination = scenario["destination"]
        
        return taxi_x * 25 + taxi_y * 5 + passenger_loc * 5 + destination

def main():
    generator = TaxiScenarioGenerator()
    scenarios = generator.generate_scenarios(num_normal=3, num_extreme=2)
    
    print("Generated Scenarios:")
    for i, scenario_data in enumerate(scenarios):
        print(f"\nScenario {i+1} ({scenario_data['type']}):")
        scenario = scenario_data['scenario']
        print(f"Taxi Position: {scenario['taxi_position']}")
        print(f"Passenger Location: {scenario['passenger_location']}")
        print(f"Destination: {scenario['destination']}")
        
        # Try to visualize the scenario
        env = generator.visualize_scenario(scenario)
        if env is not None:
            input("Press Enter to continue...")
            env.close()
        else:
            input("Press Enter to continue with next scenario...")

if __name__ == "__main__":
    main()
