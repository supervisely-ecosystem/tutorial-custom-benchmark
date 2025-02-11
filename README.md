# Custom Benchmark Integration Tutorial

This is a source code repository for the tutorial on how to integrate a custom benchmark into the Supervisely. The tutorial is available [here](https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/custom-benchmark).

Screenshot of the resulting report we will implement in this tutorial:

![result](https://github.com/supervisely-ecosystem/tutorial-custom-benchmark/releases/download/v0.0.1/benchmark_result.png)

## How to run the tutorial

1. Clone the repository

```bash
git clone git@github.com:supervisely-ecosystem/tutorial-custom-benchmark.git
```

2. Install the dependencies

```bash
pip install -r dev_requirements.txt
```

3. Run the tutorial

- To run 1st part of the tutorial, execute the following command:

```bash
python src/main_1.py
```

- To run 2nd part of the tutorial, execute the following command:

```bash
python src/main_2.py
```

- To run 3rd part of the tutorial, execute the following command:

```bash
uvicorn src.main_3:app --host 0.0.0.0 --port 8000 --ws websockets --reload
```

4. Find the resulting report link in the console output and open it in a browser.

![report_link](https://github.com/supervisely-ecosystem/tutorial-custom-benchmark/releases/download/v0.0.1/benchmark_link.png)

### Debugging

To debug the tutorial, you can run these scripts in a debug mode. For example, to run the 1st or 2nd part of the tutorial select the corresponding configuration (`Python Current File`) in `Run and Debug` panel and press `F5`.
To run the 3rd part of the tutorial in a debug mode, select the `GUI app` configuration in `Run and Debug` panel and press `F5`.
