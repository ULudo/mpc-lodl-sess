# MPC LODL SESS

This repository contains the code (src directory) and result files (out directory) accompanying the paper **“A Computationally Efficient End-to-End Learning Approach for Smart Energy Storage Systems”**, accepted at *IEEE ISGT Europe 2025*.

It provides the implementation of the **Locally Optimized Decision Loss (LODL)** framework for **Smart Energy Storage Systems (SESS)**. LODL learns a decision-aware loss function from MPC sensitivity analysis and uses it to train a recurrent forecasting model whose prediction errors are weighted according to their impact on control performance.  

The SESS environment data and the recurrent prediction model are based on the [*MPC Predictor SESS Benchmark*](https://github.com/ULudo/mpc-predictor-sess-benchmark/tree/main) repository.

**Note:** The precomputed LODL target file (`lodl_targets.pt`) could not be uploaded to this repository, as it exceeds GitHub’s file size limit.  
Please generate it locally using the provided scripts before running the experiments.

## Citation

If you use this repository or find it helpful in your own research, please cite:

> **Note:** This paper is accepted for publication and will appear in *IEEE Xplore*.

```bibtex
@inproceedings{Ludolfinger2025,
  title     = {A Computationally Efficient End-to-End Learning Approach for Smart Energy Storage Systems},
  author    = {Ludolfinger, Ulrich and Hamacher, Thomas and Martens, Maren},
  booktitle = {IEEE ISGT Europe 2025},
  year      = {2025},
  note      = {to appear}
}
```

## Licensing

* The code and documentation are released under the MIT License.
* The electricity price data in `res/data/ee_prices.csv` are © Bundesnetzagentur | SMARD.de and redistributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
* The files `opsd_building_*.csv` in `res/data` originate from the [Open Power System Data](https://data.open-power-system-data.org/household_data/2020-04-15/) project and are also redistributed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.