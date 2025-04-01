import optuna
import subprocess
import re


def objective(trial):
    # Fix the dataset as a constant, e.g., "amazon-computer"
    dataset = "roman-empire" #cora citeseer pubmed amazon-photo wikics

    # Select the number of hidden channels
    hidden_channels = trial.suggest_categorical("hidden_channels", [80, 128, 168])

    # Select the number of local layers
    local_layers = trial.suggest_categorical("local_layers", [8,9,10])

    # Select the number of training epochs
    # epochs = trial.suggest_categorical("epochs", [500])
    epochs = 2000

    # Select the learning rate
    lr = trial.suggest_loguniform("lr", 0.005, 0.01)

    # Select the weight decay
    weight_decay = trial.suggest_categorical("weight_decay", [0, 0.0001])

    # Select the dropout rate
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])

    K = trial.suggest_categorical("K", [1, 2, 3, 4, 5, 6])
    heads = trial.suggest_categorical("heads", [1, 2, 4, 6, 8])
    gamma = trial.suggest_loguniform("gamma", 0.2, 1.2)
    fea_drop =  trial.suggest_categorical("fea_drop", ["simple","normal","none"])

    base_model = trial.suggest_categorical("base_model", ['gcn', 'gat'])

    # if "gcn" in base_model:
    #     hop_att = "False"
    # else:
    #     hop_att = "True"

    hop_att = trial.suggest_categorical("hop_att", [True, False])

    # Fix the number of runs to 3
    runs = 5

    # Whether to use LayerNorm
    # use_ln = trial.suggest_categorical("use_ln", [True, False])
    use_ln =False
    ln_flag = "--ln" if use_ln else ""

    # Whether to use BatchNorm
    # use_bn = trial.suggest_categorical("use_bn", [True, False])
    use_bn = True
    bn_flag = "--bn" if use_bn else ""

    # Whether to use Residual Connections
    # use_res = trial.suggest_categorical("use_res", [True, False])
    use_res =False
    res_flag = "--res" if use_res else ""

    # Whether to use Pre-Linear layer
    # use_pre_linear = trial.suggest_categorical("use_pre_linear", [True, False])
    use_pre_linear = True
    pre_linear_flag = "--pre_linear" if use_pre_linear else ""

    # Build the command (remove empty string parameters)
    command = [
        "python", "main.py",
        "--gnn", "gens",
        "--dataset", dataset,
        "--hidden_channels", str(hidden_channels),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--runs", str(runs),
        "--local_layers", str(local_layers),
        "--weight_decay", str(weight_decay),
        "--dropout", str(dropout),
        "--K", str(K),
        "--base_model", base_model,
        "--hop_att", str(hop_att),
        "--gamma", str(gamma),
        "--fea_drop",str(fea_drop),
        "--device","0",
        "--heads",str(heads),
        # "--metric", "rocauc",
        # "--rand_split_class",
        # "--valid_num", "500",
        # "--test_num", "1000",
        # "--seed", "123"
    ]

    # Add flags only if they are non-empty
    if ln_flag:
        command.append(ln_flag)
    if bn_flag:
        command.append(bn_flag)
    if res_flag:
        command.append(res_flag)
    if pre_linear_flag:
        command.append(pre_linear_flag)

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print standard output and error output for debugging
    print("Standard Output:")
    print(result.stdout)
    print("Standard Error:")
    print(result.stderr)

    # Get the standard output
    output = result.stdout

    # Use regular expression to extract the value of param_sel
    match = re.search(r"param_sel:\s*([0-9.]+)", output)

    # If param_sel is found, return its value; otherwise return a default negative value
    if match:
        param_sel = float(match.group(1))
    else:
        print("No param_sel found in the output!")
        param_sel = 0.0  # Return a default value if param_sel is not found

    # Return the evaluation result
    return param_sel


# Create an Optuna study
study = optuna.create_study(direction="maximize")  # Assume the goal is to maximize accuracy
study.optimize(objective, n_trials=500)  # Run 500 trials

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)