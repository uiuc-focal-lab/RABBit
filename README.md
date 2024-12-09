# [Relational Verification Leaps Forward with RABBit](https://openreview.net/pdf?id=W5U3XB1C11) (NeurIPS 2024)


Official Implementation of [Relational Verification Leaps Forward with RABBit](https://openreview.net/pdf?id=W5U3XB1C11).

## Installation and Setup 

### Installing Gurobi

GUROBI installation instructions can be found at `https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer`



#### Update environment variables:
i) Run following export commands in command prompt/terminal (these environment values are only valid for the current session) 
ii) Or copy the lines in the .bashrc file (or .zshrc if using zshell), and save the file 

```
export GUROBI_HOME=/opt/gurobi1100/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:"$LD_LIBRARY_PATH"
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
```

#### Getting the free academic license:
To run GUROBI one also needs to get a free academic license. 
https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license

a) Register using any academic email ID on the GUROBI website. 
b) Generate the license on https://portal.gurobi.com/iam/licenses/request/. Choose Named-user Academic.
c) Use the command in the command prompt to generate the licesne. 


(If not automatically done, place the license in one of the following locations “/opt/gurobi/gurobi.lic” or “$HOME/gurobi.lic”)

### Downloading and Running Trained Models 
We use networks trained using both standard training methods and robust training strategies, such as [DiffAI](https://github.com/eth-sri/diffai), [SABR](https://github.com/eth-sri/SABR), and [CITRUS](https://arxiv.org/pdf/2405.09176). Our experiments utilize publicly available pre-trained DNNs sourced from the [CROWN repository](https://github.com/Verified-Intelligence/auto_LiRPA), [α, β-CROWN repository](https://github.com/Verified-Intelligence/alpha-beta-CROWN), and [ERAN repository](https://github.com/eth-sri/eran). We provide various trained networks used for the experiments in `RABBit/nets`. 

### Installing Dependencies
```
cd src/abc/
conda env create -f complete_verifier/environment.yaml --name rabbit
# activate the environment
conda activate rabbit
```

Alternatively, if the above doesn't work, refer to this [installation script](https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/8e804f0ad6f6b1726f0e916549b9477bb66f0317/vnncomp_scripts/install_tool_general.sh) from α,β-CROWN. 

## Running Experiments 

#### Caveats 
- The results obtained in the experiment can slightly vary depending on the machine, GPU and CPU load, and batch size that can fit in memory. 
- Increasing the hyperparameters $k_t$ and timeout  can improve precision.

### Instructions for Running Experiments 
Refer to`RABBit/tests/test_rabbit.py` for running experiments for RABBit. All baselines can be run from their respective open-source repositories. 

#### Hyperparameters
The following table describes key hyperparameters used by RABBit.
  | Parameter Name        | Type           | Description  |
  | ------------- |:-------------:| -----:|
  | `raven_mode`      | choices = RavenMode.UAP | The relational property being verified. For both k-UAP and top-k, use  RavenMode.UAP. |
  | `dataset`     | choices = Dataset.MNIST, Dataset.CIFAR10      | The dataset verification is run on. |
  | `verify_mode`     | str, choices = 'meta', 'topk'     | 'meta' runs k-UAP verification for RABBit. 'topk' runs  top-k verification for RABBit. |
  | `net_names`     | List[str]      | A list of size 1 of the network to be verified |
  | `device`     | str      | The torch device type.  |
  | `count_per_prop`     | int, default = 50      | Number of inputs per property $k$. |
  | `prop_count`     | int, default = 10      | Number of relational properties to run. |
  | `eps`     | float      | ϵ hyperparameter |
  | `overall_batch_size`     | int, default = 512      |   Overall batch size used for verifier. |
  | `bab_batch_size`     | int, default = 64      | Batch size used for BaB. |
  | `refinement_batch_size`     | int, default = 300      | Batch size used for cross-executional refinement. |
  | `execution_count_dct`     | dict      | The max value in the dictionary represents the $k_t$ hyperparameter. |
  | `bab_timeout`     | int, default = 60      | Timeout, in seconds, per input for BaB. Total verification time per property is computed as `bab_timeout` * `count_per_prop` |

#### UAP Verification
Examples of UAP experiments are found in 
  `test_rabbit.TestUAP`.
  
 One can run a single experiment from the test using the following command. The experiment runs k-UAP verification for RABBit for the ConvSmall network with $\epsilon$ = 2.0 and trained with SABR on the CIFAR-10 dataset for 10 properties with a count per property of 50.   
 
 `python -m unittest -v tests.test_rabbit.TestUAP.test_cifar_uap_sabr`

  
  #### Results 
  Results are stored in `results/crossex_complete`. Refer to the example test cases in `RABBit/tests/test_results.py` for analyzing results. 


## Adding New Experiments
Similar to existing experiments one can easily add new experiments using a unit test. One can add this test in existing test file `RABBit/tests/test_rabbit.py` or can create a new test file in `RABBit/tests/`.
  
 More information about the adding unittests in python is available here 
  
  https://docs.python.org/3/library/unittest.html.
 
 A test function looks like following 
 ```python
    def test_cifar_uap_diffai(self):
        for verify_mode in ['meta']:
            for prop_id in range(10):
                args = ver.RavenArgs(raven_mode=RavenMode.UAP, dataset=Dataset.CIFAR10, net_names= ['cifar10convSmallRELUDiffAI.onnx'],
                            count_per_prop=50, prop_count=10, eps=5.0/255,
                            threshold_execution=4, cross_executional_threshold=5, maximum_cross_execution_count=4, 
                            baseline_iteration=20, refinement_iterations=20, unroll_layers = False, unroll_layer_count=3, 
                            refine_intermediate_bounds =True, optimize_layers_count=2, 
                            bounds_for_individual_refinement=True, dataloading_seed = 0,
                            parallelize_executions=False, greedy_tuple= True,
                            lp_threshold=-0.5, prop_idx= prop_id,
                            max_linear_apprx=6, full_alpha= True,
                            device='cuda', prune_before_refine = 10,
                            always_correct_cross_execution = False,
                            result_dir='results_new', write_file=True, complete_verification= True, use_lp_bab= False, verify_mode = verify_mode,
                            execution_count_dct = {2: 10 ,3: 10, 4:8, 5:8},  bias = 0.1, use_ib_refinement = True,  bab_timeout = 60, max_targetted= 4, 
                            num_targetted= 10, refinement_batch_size= 600,
                            store_time_trace= True, branching_method= 'babsr')    
                ver.adptiveRaven(raven_args=args)   
```



 
