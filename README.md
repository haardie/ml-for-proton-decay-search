
# Machine Learning for Proton Decay Search at DUNE
The code in this repository supplements the work presented in my bachelor thesis, which I have yet to defend.

I strived to make the code as human-readable as possible and at least a bit modular for overall simplicity and a more pleasant workflow.
Here, I describe the repository structure.

There are three utility files: `cls.py`, `fns.py` and `net_cfg.json`.

- `cls.py` contains the class constructors of different functionalities: the sparse matrix datasets, modifications to the known architectures as well as custom ones;
- In `fns.py`, the functions utilized at all stages of the data processing pipeline are defined;
- Finally, `net_cfg.json` is a config file containing the NN hyperparameters as well as space for a little bit more description (which is useful for logging).

There are three main training-focused files: `train_single_plane.py` for training the modified ResNet18 models, `late.py` for the late fusion approach, and `early.py` for the early fusion approach. The `hyper_search.py` and the `wandb_sweep.py` files were used for the hyperparameter search and tuning.

There is plenty of code for the miscellaneous functionality:
- signal/background distributions plotting,
- the calculation of the normalization parameters,
- attempts to assess model performance using the MC dropout inference technique,
- reading and simplifying the event generator logs and sorting the files based on the $K^+$ decay channel

And I guess I would have found more on my local machine :)

Worth noting is the image-preprocessing files: those are `sparsify_csv.py`, `ROI_functions.py`, and `find2lines.py`. Their functionality is as follows:

- `ROI_functions.py` is a utility file containing the functions for ROI extraction and centering;
- `sparsify_csv.py` extracts the ROI from the CSV file and saves it as a CSR sparse matrix;
- `find2lines.py` is a constrained flood fill routine for the detection of the *double-track* instances.

When it comes to documentation, most of the code is supplemented with comments, and the functions are ensured to have the *docstrings*. In the future, I plan to create more extensive documentation and, of course, refine the existing code.

With regards,
Anna
