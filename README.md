# CO3-skitour-optimization
This project compares **Dijkstra’s algorithm** and **Simulated Annealing** for ski tour route optimization on terrain data. The focus is on computation time, path quality, and scalability with increasing distance.

## Usage
Run from project root:

```sh
pip install -e .
``` 

Then run the individual scripts such as:

```sh
python scripts/efficiency.py
```

to obtain its results:

![Alt text](data/results/results_plots/sa_vs_dijk_computation_and_path_time_plot.png)

and 

![Alt text](data/results/results_plots/sa_vs_dijk_path_plot_for_appendix.png)

and similarly you can run:

```sh
python scripts/accuracy.py
python scripts/plot_comparison_3.py
python scripts/robustness.py
python scripts/sa_parameter_search.py
```


