# Experiments
Run these Python scripts in the terminal with `-h` to see their arguments.
## Running experiments
- `main.py`: Main experimetnation script.
- `BrainOptimizer.py`: Differential Evolution implementation for controller learning.
- `ModularEvolution.py`: Classes for the various steps of brain/body evolution (parent/survivor selection, crossover/mutation).
- `evaluator_brain_targeted_locomotion.py`: Evaluator class to run simulations and retrieve fitnesses.
- `config.py`: Parameters used for running experiments.
- `run_deconstructed.py`: Decomposed experiment for debugging and analysis with e.g. variable explorers.
- `consolePlot.py`: Plot a generation's fitnesses in the console (called in `main.py` etc.).
- `writeOut.py`: Writes settings to output file when running experiments.
- `/Databases`: Experiment databases are stored in here.

## Analysis
- `rerun.py`: Render or plot the trajectory for the best robot in a database.
- `rerun_multiple.py`: Plot the best robot per experiment for databases in a supplied list. $n$ best robots can be chosen.
- `BodyCheck.py`: (often called as morpho in scripts) Analyse robot bodies, create morpho feature vectors etc.
- `plot.py`: Plot the mean and max fitnesses in a databse.
- `combinedPlot.py`: Plot the mean and max fitnesses for all databases in a supplied list.
- `generate_df.py`: Generate mean and max fitness dataframes for all databases in a supplied list.
- `morphAnalysis.py`: Calculate and plot novelty and diversity. Also exports dataframes for statistical analysis.
- `morphoVis.py`: Generate 2D plots for warm and cold start databases in supplied lists.
- `db_cold.json & db_warm.json`: JSON files containing the names databases in `/Databases` to use in some of the above scripts.
