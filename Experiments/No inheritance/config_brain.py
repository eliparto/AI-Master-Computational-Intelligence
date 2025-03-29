"""Configuration parameters for this example."""

from revolve2.standards.modular_robots_v2 import gecko_v2

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 4
NUM_SIMULATORS = 4
INITIAL_STD = 0.5
NUM_GENERATIONS = 5
BODY = gecko_v2()
