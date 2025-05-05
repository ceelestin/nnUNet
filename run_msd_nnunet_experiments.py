import os
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, RepeatedKFold
from pathlib import Path
import glob # Added for file scanning
import json # Added for results parsing
import tempfile # Added for temporary directories
import shutil # Added for file operations
import SimpleITK as sitk # Added for Dice calculation

# --- Configuration ---
# nn-UNet Environment Variables (ensure these are set in your environment)
NNUNET_RAW_DATA_BASE = Path(os.environ.get("nnUNet_raw_data_base", "/path/to/default/nnUNet_raw_data_base"))
NNUNET_PREPROCESSED = Path(os.environ.get("nnUNet_preprocessed", "/path/to/default/nnUNet_preprocessed"))
RESULTS_FOLDER = Path(os.environ.get("RESULTS_FOLDER", "/path/to/default/nnUNet_trained_models"))
NNUNET_V1_EXECUTABLE_PATH = "nnUNet_train" # Adjust if nnUNet commands are not directly in PATH
NNUNET_FIND_BEST_CONFIG_PATH = "nnUNet_find_best_configuration"
NNUNET_PREDICT_PATH = "nnUNet_predict"

# Ensure base directories exist (optional, nnUNet might create them)
NNUNET_RAW_DATA_BASE.mkdir(parents=True, exist_ok=True)
NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# MSD Task Information
# IMPORTANT: Ensure these Task Names EXACTLY match your nnUNet folder names
TASK_INFO = {
    "Task001_BrainTumour": {"id": "1", "n_samples": 484, "n_biggest_study": 67},
    # "Task002_Heart": {"id": "2", "n_samples": 20, "n_biggest_study": 3},
    # "Task003_Liver": {"id": "3", "n_samples": 131, "n_biggest_study": 18},
    # "Task004_Hippocampus": {"id": "4", "n_samples": 260, "n_biggest_study": 36},
    # "Task005_Prostate": {"id": "5", "n_samples": 32, "n_biggest_study": 4},
    # "Task006_Lung": {"id": "6", "n_samples": 63, "n_biggest_study": 9},
    # "Task007_Pancreas": {"id": "7", "n_samples": 281, "n_biggest_study": 39},
    # "Task008_HepaticVessel": {"id": "8", "n_samples": 303, "n_biggest_study": 42},
    # "Task009_Spleen": {"id": "9", "n_samples": 41, "n_biggest_study": 6},
    # "Task010_Colon": {"id": "10", "n_samples": 126, "n_biggest_study": 17},
}
TOTAL_SAMPLES = sum(info["n_samples"] for info in TASK_INFO.values())
TOTAL_BIGGEST_STUDY = sum(info["n_biggest_study"] for info in TASK_INFO.values())
TOTAL_BENCHMARKING = TOTAL_SAMPLES - TOTAL_BIGGEST_STUDY

print(f"Total Labeled Samples: {TOTAL_SAMPLES}")
print(f"Target Biggest Study Set Size: {TOTAL_BIGGEST_STUDY}")
print(f"Target Benchmarking Set Size: {TOTAL_BENCHMARKING}")

# Splitting Parameters
INITIAL_SPLIT_SEED = 42
STUDY_SET_SIZES_PERC = [0.10, 0.16, 0.25, 0.40, 0.63, 1.00] # 10%, 16%, ..., 100%
STUDY_SET_REPETITIONS = 1 # Seeds for the second split
CV_TEST_SIZE = 0.2
CV_N_SPLITS = 5 # For KFold and ShuffleSplit base

# Cross-Validation Configurations
CV_CONFIGS = [
    {"type": "TrainTestSplit", "params": {"test_size": CV_TEST_SIZE}},
    {"type": "ShuffleSplit_5", "params": {"n_splits": 5, "test_size": CV_TEST_SIZE}},
    {"type": "ShuffleSplit_10", "params": {"n_splits": 10, "test_size": CV_TEST_SIZE}},
    {"type": "RepeatedKFold_5_2", "params": {"n_splits": CV_N_SPLITS, "n_repeats": 2}},
    {"type": "RepeatedKFold_5_3", "params": {"n_splits": CV_N_SPLITS, "n_repeats": 3}},
    {"type": "RepeatedKFold_5_5", "params": {"n_splits": CV_N_SPLITS, "n_repeats": 5}},
    {"type": "RepeatedKFold_5_10", "params": {"n_splits": CV_N_SPLITS, "n_repeats": 10}},
]

# --- Helper Functions ---

def get_actual_sample_ids(task_name, expected_n_samples):
    """Scans the nnUNet raw data folder for actual sample IDs."""
    task_raw_data_dir = NNUNET_RAW_DATA_BASE / "nnUNet_raw_data" / task_name / "imagesTr"
    if not task_raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found for task {task_name}: {task_raw_data_dir}")

    # Find files like 'CASE_ID_0000.nii.gz' and extract 'CASE_ID'
    image_files = glob.glob(str(task_raw_data_dir / "*_0000.nii.gz"))
    sample_ids = sorted([Path(f).name.split('_0000.nii.gz')[0] for f in image_files])

    if len(sample_ids) != expected_n_samples:
        print(f"Warning: Found {len(sample_ids)} samples for {task_name}, but expected {expected_n_samples}. Using found samples.")
        # Update TASK_INFO if needed, or handle discrepancy
        # TASK_INFO[task_name]["n_samples"] = len(sample_ids) # Be careful with this

    if not sample_ids:
         raise ValueError(f"No samples found for task {task_name} in {task_raw_data_dir}")

    return sample_ids

def create_custom_split_file(filepath, train_ids, val_ids):
    """Creates a splits_final.pkl file for nnUNet."""
    splits = [{'train': train_ids, 'val': val_ids}]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(splits, f)
    print(f"Created split file: {filepath}")

def run_command(command_list, log_file):
    """Runs a shell command and logs output."""
    print(f"Executing: {' '.join(command_list)}")
    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            process = subprocess.run(
                command_list,
                stdout=f,
                stderr=subprocess.STDOUT, # Redirect stderr to stdout
                text=True,
                check=True # Raise exception on non-zero exit code
            )
        end_time = time.time()
        duration = end_time - start_time
        print(f"Command finished successfully in {duration:.2f} seconds.")
        return duration, True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Command failed after {duration:.2f} seconds. Error: {e}")
        print(f"Check log file: {log_file}")
        return duration, False
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"An unexpected error occurred: {e}")
        return duration, False

def parse_nnunet_results(json_path):
    """Parses the summary.json file to get Dice score."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # nnUNet often has results per class. Assuming foreground class is '1'.
        # Adjust the keys based on your actual summary.json structure
        dice = data['results']['mean'].get('1', {}).get('Dice')
        if dice is None:
             # Fallback: Check if maybe foreground class is not '1' or structure differs
             # Example: Try getting the first available class metric if '1' is missing
             foreground_classes = [k for k in data['results']['mean'].keys() if k != '0'] # Exclude background
             if foreground_classes:
                 dice = data['results']['mean'][foreground_classes[0]].get('Dice')

        if dice is not None:
            print(f"  Successfully parsed Dice from {json_path}: {dice:.4f}")
            return dice
        else:
            print(f"  Warning: Could not find foreground Dice score in {json_path}. Structure might be unexpected.")
            return None
    except FileNotFoundError:
        print(f"  Error: Results file not found: {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON from {json_path}")
        return None
    except KeyError as e:
        print(f"  Error: Key missing in JSON structure ({e}) in {json_path}")
        return None
    except Exception as e:
        print(f"  Error parsing results file {json_path}: {e}")
        return None

def calculate_dice_simpleitk(pred_path, gt_path, label_class=1):
    """Calculates Dice score using SimpleITK for a specific class."""
    if not pred_path.exists() or not gt_path.exists():
        print(f"  Warning: Prediction or GT file missing for Dice calculation. Pred: {pred_path}, GT: {gt_path}")
        return None

    try:
        pred_img = sitk.ReadImage(str(pred_path))
        gt_img = sitk.ReadImage(str(gt_path))

        pred_arr = sitk.GetArrayFromImage(pred_img)
        gt_arr = sitk.GetArrayFromImage(gt_img)

        # Ensure arrays are boolean for the specified class
        pred_bool = (pred_arr == label_class)
        gt_bool = (gt_arr == label_class)

        intersection = np.logical_and(pred_bool, gt_bool).sum()
        total_sum = pred_bool.sum() + gt_bool.sum()

        if total_sum == 0: # Both prediction and GT are empty for this class
            return 1.0 # Perfect score according to some definitions
        else:
            dice = (2. * intersection) / total_sum
            return dice

    except Exception as e:
        print(f"  Error calculating Dice for {pred_path.name}: {e}")
        return None


def evaluate_on_benchmarking_set(model_output_dir, task_name, benchmarking_samples, run_dir, nnunet_raw_data_base, nnunet_predict_path):
    """
    Runs prediction on benchmarking samples and calculates Dice score.
    """
    print(f"  Evaluating model from {model_output_dir} on benchmarking set for task {task_name}...")

    # 1. Identify benchmarking samples for the current task
    # Assumes sample IDs start with something identifiable like 'TaskXXX_' - adjust if needed
    task_prefix = task_name.split('_')[0] # e.g., Task001
    task_benchmarking_samples = [s for s in benchmarking_samples if s.startswith(task_prefix)]

    if not task_benchmarking_samples:
        print("    No benchmarking samples found for this task.")
        return None

    print(f"    Found {len(task_benchmarking_samples)} benchmarking samples for {task_name}.")

    # 2. Create temporary directories
    with tempfile.TemporaryDirectory(suffix="_bench_img_in", prefix=f"{run_dir.name}_") as temp_img_dir_str,\
         tempfile.TemporaryDirectory(suffix="_bench_lbl_in", prefix=f"{run_dir.name}_") as temp_lbl_dir_str,\
         tempfile.TemporaryDirectory(suffix="_bench_pred_out", prefix=f"{run_dir.name}_") as temp_pred_dir_str:

        temp_img_dir = Path(temp_img_dir_str)
        temp_lbl_dir = Path(temp_lbl_dir_str)
        temp_pred_dir = Path(temp_pred_dir_str)
        print(f"    Staging benchmarking data in temporary directories...")

        # 3. Copy/link image and label files
        missing_files = False
        for sample_id in task_benchmarking_samples:
            # Assuming benchmark samples originated from the 'Tr' (training) sets
            img_suffix = "_0000.nii.gz"
            lbl_suffix = ".nii.gz"
            src_img_path = nnunet_raw_data_base / "nnUNet_raw_data" / task_name / "imagesTr" / f"{sample_id}{img_suffix}"
            src_lbl_path = nnunet_raw_data_base / "nnUNet_raw_data" / task_name / "labelsTr" / f"{sample_id}{lbl_suffix}"

            dest_img_path = temp_img_dir / f"{sample_id}{img_suffix}"
            dest_lbl_path = temp_lbl_dir / f"{sample_id}{lbl_suffix}" # Store labels for later evaluation

            if src_img_path.exists() and src_lbl_path.exists():
                try:
                    # Using copy instead of symlink for simplicity, adjust if large files/performance issue
                    shutil.copy(src_img_path, dest_img_path)
                    shutil.copy(src_lbl_path, dest_lbl_path)
                except Exception as e:
                    print(f"    Error copying files for sample {sample_id}: {e}")
                    missing_files = True
                    break # Stop if copying fails
            else:
                print(f"    Warning: Missing source image or label for benchmarking sample {sample_id}")
                print(f"      Img Path: {src_img_path}")
                print(f"      Lbl Path: {src_lbl_path}")
                missing_files = True
                # Continue to copy others, but evaluation might be incomplete

        if missing_files and len(list(temp_img_dir.glob('*'))) == 0:
             print("    Error: No valid benchmarking images could be staged. Aborting evaluation.")
             return None # Cannot proceed if no images were copied

        # 4. Run nnUNet_predict
        predict_bench_log = run_dir / "predict_bench.log"
        predict_bench_cmd = [
            nnunet_predict_path,
            "-i", str(temp_img_dir),
            "-o", str(temp_pred_dir),
            "-t", task_name,
            "-m", "3d_fullres",
            "-f", "0", # Assuming fold 0 model from training
            # '-chk', 'model_final_checkpoint', # Specify checkpoint if needed
            "--save_npz" # Optional: save softmax, not needed for Dice
        ]

        print(f"    Running prediction on {len(task_benchmarking_samples)} benchmarking samples...")
        duration, success = run_command(predict_bench_cmd, predict_bench_log)

        if not success:
            print(f"    Prediction on benchmarking set failed. Check log: {predict_bench_log}")
            return None

        # 5. Calculate Dice scores
        print(f"    Calculating Dice scores for predictions in {temp_pred_dir}...")
        dice_scores = []
        for sample_id in task_benchmarking_samples:
            pred_file = temp_pred_dir / f"{sample_id}.nii.gz"
            gt_file = temp_lbl_dir / f"{sample_id}.nii.gz" # Ground truth copied earlier

            if not pred_file.exists():
                print(f"    Warning: Prediction file missing for sample {sample_id}. Skipping Dice calculation.")
                continue

            # Calculate Dice for foreground class 1 (adjust if needed)
            dice = calculate_dice_simpleitk(pred_file, gt_file, label_class=1)
            if dice is not None:
                dice_scores.append(dice)
            else:
                # Logged within calculate_dice_simpleitk
                pass

        if not dice_scores:
            print("    Error: No Dice scores could be calculated for the benchmarking set.")
            return None

        average_dice = np.mean(dice_scores)
        print(f"    Average Dice score on benchmarking set ({len(dice_scores)} samples): {average_dice:.4f}")

        # 6. Cleanup is handled by 'with tempfile.TemporaryDirectory...'
        print("    Benchmarking evaluation complete.")
        return average_dice

# --- Main Script ---

# 1. Generate Full Dataset Representation
all_samples = []
all_task_labels = []
task_samples_dict = {}
print("Attempting to load actual sample IDs...")
try:
    for task_name, info in TASK_INFO.items():
        # Use the new function to get actual IDs
        samples = get_actual_sample_ids(task_name, info["n_samples"])
        # Update n_samples if it differed from expectation (optional, depends on desired behavior)
        if len(samples) != info["n_samples"]:
             print(f"Updating sample count for {task_name} from {info['n_samples']} to {len(samples)}")
             TASK_INFO[task_name]["n_samples"] = len(samples)
             # Recalculate proportional biggest study size based on actual samples?
             # This might be complex if initial proportions must be strictly kept.
             # For now, we proceed with the original target counts for splitting,
             # but use the actual sample list. This might lead to minor deviations
             # if sample counts changed significantly.
             # Re-calculate total samples based on actual findings
             # TOTAL_SAMPLES = sum(t_info["n_samples"] for t_info in TASK_INFO.values())
             # print(f"Adjusted Total Labeled Samples: {TOTAL_SAMPLES}")


        task_samples_dict[task_name] = samples
        all_samples.extend(samples)
        all_task_labels.extend([task_name] * len(samples)) # Use actual length
except (FileNotFoundError, ValueError, Exception) as e:
    print(f"\nError loading sample IDs: {e}")
    print("Please ensure nnUNet environment variables are set correctly and raw data exists.")
    exit(1) # Exit if we can't get sample IDs

# Re-calculate totals based on potentially updated counts
TOTAL_SAMPLES = sum(info["n_samples"] for info in TASK_INFO.values())
# Keep target study counts as initially defined, split will handle discrepancies
# TOTAL_BIGGEST_STUDY = sum(info["n_biggest_study"] for info in TASK_INFO.values())
TOTAL_BENCHMARKING = TOTAL_SAMPLES - TOTAL_BIGGEST_STUDY # This might differ slightly now

print(f"\nActual Total Labeled Samples Found: {TOTAL_SAMPLES}")
print(f"Target Biggest Study Set Size: {TOTAL_BIGGEST_STUDY}")
print(f"Resulting Benchmarking Set Size: {TOTAL_BENCHMARKING}")

# 2. Initial Split: Biggest Study Set vs. Benchmarking Set
print(f"\nPerforming initial split (Seed: {INITIAL_SPLIT_SEED})...")
# We need to split each task individually to guarantee proportions
biggest_study_set_samples = []
benchmarking_set_samples = []
biggest_study_set_tasks = []
benchmarking_set_tasks = []

for task_name, info in TASK_INFO.items():
    task_all_samples = task_samples_dict[task_name]
    n_biggest = info["n_biggest_study"]
    n_total = info["n_samples"]

    if n_biggest > 0 and n_total > 0:
        if n_biggest >= n_total: # Handle cases where study set is all samples
             study_task, benchmark_task = task_all_samples, []
        elif n_biggest == 0:
             study_task, benchmark_task = [], task_all_samples
        else:
            study_task, benchmark_task = train_test_split(
                task_all_samples,
                test_size=(n_total - n_biggest), # Size of benchmarking set for this task
                random_state=INITIAL_SPLIT_SEED,
                # No stratification needed here as we operate per task
            )
        biggest_study_set_samples.extend(study_task)
        benchmarking_set_samples.extend(benchmark_task)
        biggest_study_set_tasks.extend([task_name] * len(study_task))
        benchmarking_set_tasks.extend([task_name] * len(benchmark_task))
    elif n_total > 0: # Only benchmarking samples
        benchmarking_set_samples.extend(task_all_samples)
        benchmarking_set_tasks.extend([task_name] * len(task_all_samples))


print(f"  Biggest Study Set size: {len(biggest_study_set_samples)}")
print(f"  Benchmarking Set size: {len(benchmarking_set_samples)}")
# Verification (optional)
# print("Biggest Study Set Counts per Task:")
# print(pd.Series(biggest_study_set_tasks).value_counts())

# --- Prepare for Experiment Loop ---
results = []
experiment_commands = []
# <<< SET TO True TO RUN nnUNet NOW (with caveats) >>>
# <<< Ensure you have implemented benchmarking evaluation if True >>>
EXECUTE_COMMANDS_DIRECTLY = True
BASE_EXPERIMENT_DIR = Path("./msd_nnunet_experiments")
BASE_EXPERIMENT_DIR.mkdir(exist_ok=True)

# 3. Loop through Study Set Sizes and Repetitions
for size_perc in STUDY_SET_SIZES_PERC:
    actual_study_set_target_size = int(round(TOTAL_BIGGEST_STUDY * size_perc))
    print(f"\n--- Processing Study Set Size: {size_perc*100:.0f}% ({actual_study_set_target_size} samples) ---")

    for rep_seed in range(STUDY_SET_REPETITIONS):
        print(f"  Repetition Seed: {rep_seed}")

        # 4. Second Split: Actual Study Set vs. Leftout Set (from Biggest Study Set)
        if size_perc == 1.0:
            # If 100%, the actual study set is the entire biggest study set
            actual_study_set_samples = biggest_study_set_samples
            actual_study_set_tasks = biggest_study_set_tasks
            # leftout_set_samples = [] # Not strictly needed later
        else:
            # Stratify this split by task within the biggest study set
            actual_study_set_samples, _leftout_samples, actual_study_set_tasks, _leftout_tasks = train_test_split(
                biggest_study_set_samples,
                biggest_study_set_tasks,
                train_size=size_perc, # Use percentage directly
                random_state=rep_seed,
                stratify=biggest_study_set_tasks
            )

        print(f"    Actual Study Set size for this rep: {len(actual_study_set_samples)}")
        # Convert to list for consistency if needed later (train_test_split returns lists)
        actual_study_set_samples = list(actual_study_set_samples)
        actual_study_set_tasks = list(actual_study_set_tasks)

        # Create a dictionary mapping task to samples for the current actual study set
        current_study_set_by_task = {}
        for sample, task in zip(actual_study_set_samples, actual_study_set_tasks):
            if task not in current_study_set_by_task:
                current_study_set_by_task[task] = []
            current_study_set_by_task[task].append(sample)

        # 5. Loop through Cross-Validation Strategies for the *current* Actual Study Set
        for cv_config in CV_CONFIGS:
            cv_type = cv_config["type"]
            cv_params = cv_config["params"]
            cv_seed = rep_seed # Use the same seed for CV splitting consistency
            print(f"      Applying CV Strategy: {cv_type} (Seed: {cv_seed})")

            # --- Generate Train/Test splits based on CV Strategy ---
            # We need to generate splits *within each task* present in the actual_study_set

            task_splits = {} # {task_name: [(train_indices, test_indices), ...]}

            for task_name, task_samples in current_study_set_by_task.items():
                if len(task_samples) < 2: # Cannot split if less than 2 samples
                    print(f"        Skipping CV for {task_name} in {cv_type} (only {len(task_samples)} sample)")
                    continue
                if cv_type == "TrainTestSplit":
                     # Ensure test_size doesn't result in 0 test samples
                    test_size = cv_params['test_size']
                    if int(len(task_samples) * test_size) < 1:
                        print(f"        Adjusting test size for {task_name} in {cv_type} to ensure at least 1 test sample.")
                        test_size = 1 / len(task_samples) # Ensure at least 1 sample
                    if int(len(task_samples) * (1-test_size)) < 1:
                         print(f"        Skipping CV for {task_name} in {cv_type} (cannot make train/test split).")
                         continue

                    try:
                        train_idx, test_idx = train_test_split(
                            range(len(task_samples)), # Split indices
                            test_size=test_size,
                            random_state=cv_seed,
                            # No stratification needed here (already per-task)
                        )
                        task_splits[task_name] = [(train_idx, test_idx)]
                    except ValueError as e:
                        print(f"        Skipping CV for {task_name} in {cv_type} due to split error: {e}")
                        continue


                elif "ShuffleSplit" in cv_type:
                    if len(task_samples) < cv_params['n_splits']: # Check if enough samples for n_splits
                         # Maybe adjust n_splits or skip? For now, skip.
                         if len(task_samples) < 2: # Cannot split at all
                              print(f"        Skipping CV for {task_name} in {cv_type} (only {len(task_samples)} sample)")
                              continue
                         else: # Try with fewer splits if possible
                              print(f"        Warning: Not enough samples in {task_name} for {cv_params['n_splits']} splits. Trying with {len(task_samples)-1} splits.")
                              cv_params_adjusted = cv_params.copy()
                              cv_params_adjusted['n_splits'] = max(1, len(task_samples)-1) # Ensure at least 1 split if possible
                              if cv_params_adjusted['n_splits'] == 1 and len(task_samples) < 2: continue # Still not possible

                              ss = ShuffleSplit(random_state=cv_seed, **cv_params_adjusted)
                              try:
                                task_splits[task_name] = list(ss.split(task_samples))
                              except ValueError as e:
                                  print(f"        Skipping CV for {task_name} in {cv_type} (adjusted) due to split error: {e}")
                                  continue

                    else:
                        ss = ShuffleSplit(random_state=cv_seed, **cv_params)
                        try:
                            task_splits[task_name] = list(ss.split(task_samples))
                        except ValueError as e:
                            print(f"        Skipping CV for {task_name} in {cv_type} due to split error: {e}")
                            continue


                elif "RepeatedKFold" in cv_type:
                    if len(task_samples) < cv_params['n_splits']:
                        print(f"        Skipping CV for {task_name} in {cv_type} (samples < n_splits)")
                        continue
                    rkf = RepeatedKFold(random_state=cv_seed, **cv_params)
                    task_splits[task_name] = []
                    # RKF gives K splits per repeat. We treat each fold as a test set once.
                    try:
                        for train_idx_k, test_idx_k in rkf.split(task_samples):
                            # Each iteration of rkf.split gives one fold definition
                            # We use the test_idx_k directly as our test set (approx 1/n_splits size)
                            # And train_idx_k as the training set
                            task_splits[task_name].append((train_idx_k, test_idx_k))
                    except ValueError as e:
                        print(f"        Skipping CV for {task_name} in {cv_type} due to split error: {e}")
                        continue

            # --- Process each split for each task ---
            split_counter = 0
            for task_name, splits_for_task in task_splits.items():
                task_id_str = TASK_INFO[task_name]["id"]
                task_samples_list = current_study_set_by_task[task_name] # Get samples for this task

                for fold_idx, (train_indices, test_indices) in enumerate(splits_for_task):
                    split_counter += 1
                    train_samples = [task_samples_list[i] for i in train_indices]
                    test_samples = [task_samples_list[i] for i in test_indices]

                    if not train_samples or not test_samples:
                        print(f"        Skipping fold {fold_idx} for {task_name} in {cv_type} due to empty train/test set after split.")
                        continue

                    # 6. Generate Commands and Prepare Execution
                    # Unique identifier for this specific run
                    run_id = f"task{task_id_str}_size{int(size_perc*100)}p_seed{rep_seed}_cv{cv_type}_fold{fold_idx}"
                    run_dir = BASE_EXPERIMENT_DIR / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)

                    # Define paths
                    split_pkl_path = run_dir / f"splits_final_{run_id}.pkl"
                    model_output_dir = RESULTS_FOLDER / "nnUNet" / "3d_fullres" / task_name / f"nnUNetTrainerV2__nnUNetPlansv2.1_{run_id}" # Custom trainer name
                    train_log_path = run_dir / "train.log"
                    predict_test_log_path = run_dir / "predict_test.log"
                    predict_bench_log_path = run_dir / "predict_bench.log"
                    results_json_path = model_output_dir / "fold_0" / "validation_raw" / "summary.json" # nnUNet default path

                    # Create the custom split file for nnUNet (using fold 0 convention)
                    create_custom_split_file(split_pkl_path, train_samples, test_samples)

                    # Generate nnUNet Commands
                    # a) Training command (using the custom split file via trainer name)
                    #    We trick nnUNet by naming the trainer uniquely and placing the split file
                    #    in the expected preprocessed location *before* training.
                    #    NOTE: This requires placing the generated .pkl file correctly.
                    #    A safer nnUNetV1 approach might be to modify the code or use a fork
                    #    that accepts split files directly. Let's try the standard way first.
                    #    We need to copy our generated split file to the *actual* preprocessed dir
                    #    before training, named 'splits_final.pkl'. This is risky if runs overlap.
                    #    Alternative: Generate a script that *first* copies the correct pkl, *then* runs train.

                    # Let's store commands and necessary setup steps instead of direct execution.
                    preprocessed_task_dir = NNUNET_PREPROCESSED / task_name
                    target_split_pkl = preprocessed_task_dir / "splits_final.pkl"

                    # Command sequence for this run
                    setup_command = f"cp '{split_pkl_path}' '{target_split_pkl}'"
                    train_command = [
                        NNUNET_V1_EXECUTABLE_PATH, "3d_fullres", "nnUNetTrainerV2", task_name, "0", # Fold 0
                        "--deterministic" # For reproducibility if desired
                        # Add '-p nnUNetPlansv2.1' if not default
                        # Add '--npz' if saving preprocessed data as npz was done
                    ]
                    # We don't use a custom trainer name here, we rely on overwriting the split file
                    # find_best_config_command = [NNUNET_FIND_BEST_CONFIG_PATH, "-t", task_id_str] # Task ID or Name? Check nnUNet docs
                    # Predict on Test Set (using the 'val' set from our split)
                    predict_test_command = [
                        NNUNET_PREDICT_PATH,
                        "-i", str(NNUNET_RAW_DATA_BASE / "nnUNet_raw_data" / task_name / "imagesTs"), # Input folder for test images
                        "-o", str(run_dir / "predictions_test"), # Output folder for predictions
                        "-t", task_name,
                        "-m", "3d_fullres",
                        "-f", "0", # Fold 0
                        "--save_npz" # Save softmax probabilities
                        # Add '-chk model_final_checkpoint' or other checkpoint name if needed
                    ]
                    # Predict on Benchmarking Set
                    # NOTE: nnUNet predict usually takes ONE input folder. Predicting on the
                    # benchmarking set might require temporarily organizing those files or
                    # running predict multiple times if they are in different task folders.
                    # This needs clarification based on how benchmarking samples are stored.
                    # Assuming they are identifiable within the original Task folders' imagesTs:
                    # We might need a post-processing step to calculate Dice only for benchmarking samples.
                    # For now, generate a placeholder command or note the complexity.
                    predict_bench_command_note = f"# Predict on Benchmarking Set - Requires identifying benchmark samples in {NNUNET_RAW_DATA_BASE / 'nnUNet_raw_data' / task_name / 'imagesTs'} and calculating metrics post-prediction"


                    command_info = {
                        "run_id": run_id,
                        "task_name": task_name,
                        "task_id": task_id_str,
                        "size_perc": size_perc,
                        "rep_seed": rep_seed,
                        "cv_type": cv_type,
                        "fold_idx": fold_idx,
                        "n_train_samples": len(train_samples),
                        "n_test_samples": len(test_samples),
                        "split_pkl_path": str(split_pkl_path),
                        "target_split_pkl_path": str(target_split_pkl),
                        "model_output_dir": str(model_output_dir),
                        "setup_command": setup_command,
                        "train_command": " ".join(train_command),
                        "predict_test_command": " ".join(predict_test_command),
                        "predict_bench_note": predict_bench_command_note,
                        "results_json_path": str(results_json_path),
                        "train_log": str(train_log_path),
                        "predict_test_log": str(predict_test_log_path),
                        "predict_bench_log": str(predict_bench_log_path),
                    }
                    experiment_commands.append(command_info)

                    # (Optional) Direct Execution Block
                    if EXECUTE_COMMANDS_DIRECTLY:
                        print(f"--- Executing Run: {run_id} ---")
                        # 1. Setup: Copy split file
                        print(f"Running setup: {setup_command}")
                        try:
                            # Use shell=True carefully, ensure paths are controlled. Add check=True.
                            subprocess.run(setup_command, shell=True, check=True, text=True, capture_output=True)
                            print("  Setup command successful.")
                        except subprocess.CalledProcessError as e:
                            print(f"  Setup failed: {e}")
                            print(f"  Stderr: {e.stderr}")
                            print(f"  Stdout: {e.stdout}")
                            results.append({**command_info, "status": "setup_failed", "train_time": None, "predict_test_time": None, "predict_bench_time": None, "dice_test": None, "dice_bench": None})
                            continue # Skip to next fold
                        except Exception as e:
                            print(f"  An unexpected error occurred during setup: {e}")
                            results.append({**command_info, "status": "setup_failed_unexpected", "train_time": None, "predict_test_time": None, "predict_bench_time": None, "dice_test": None, "dice_bench": None})
                            continue # Skip to next fold

                        # 2. Train
                        train_time, success = run_command(train_command, train_log_path)
                        if not success:
                            # Error message already printed by run_command
                            results.append({**command_info, "status": "train_failed", "train_time": train_time, "predict_test_time": None, "predict_bench_time": None, "dice_test": None, "dice_bench": None})
                            continue

                        # 3. Find Best Config (if needed for v1 inference - check docs)
                        # print("Running find best configuration...")
                        # _, success = run_command(find_best_config_command, run_dir / "find_best_config.log")
                        # if not success: # Maybe non-critical?
                        #     print("  Warning: find_best_configuration failed.")

                        # 4. Predict on Test Set
                        predict_start_time = time.time()
                        _, success = run_command(predict_test_command, predict_test_log_path)
                        predict_test_time = time.time() - predict_start_time
                        if not success:
                            results.append({**command_info, "status": "predict_test_failed", "train_time": train_time, "predict_test_time": predict_test_time, "predict_bench_time": None, "dice_test": None, "dice_bench": None})
                            continue

                        # 5. Calculate Dice on Test Set (Validation Set in nnUNet terms)
                        dice_test = parse_nnunet_results(results_json_path)


                        # 6. Predict & Calculate Dice on Benchmarking Set
                        predict_bench_start_time = time.time()
                        # Pass necessary paths to the evaluation function
                        dice_bench = evaluate_on_benchmarking_set(
                            model_output_dir,
                            task_name,
                            benchmarking_set_samples, # Full list of benchmark samples
                            run_dir,
                            NNUNET_RAW_DATA_BASE, # Pass base path
                            NNUNET_PREDICT_PATH # Pass predictor path
                        )
                        predict_bench_time = time.time() - predict_bench_start_time


                        results.append({
                            **command_info,
                            "status": "completed" if dice_test is not None else "completed_no_dice",
                            "train_time": train_time,
                            "predict_test_time": predict_test_time,
                            "predict_bench_time": predict_bench_time, # Includes prediction + Dice calculation
                            "dice_test": dice_test,
                            "dice_bench": dice_bench # Now holds the calculated score or None
                        })
                        print(f"--- Run {run_id} Finished ---")
                    # End Optional Execution Block

            print(f"      Generated {split_counter} train/test configurations for {cv_type}.")


# --- Save Results/Commands ---
commands_df = pd.DataFrame(experiment_commands)
commands_csv_path = BASE_EXPERIMENT_DIR / "experiment_commands.csv"
commands_df.to_csv(commands_csv_path, index=False)
print(f"\nSaved all experiment commands and configurations to: {commands_csv_path}")

if EXECUTE_COMMANDS_DIRECTLY:
    results_df = pd.DataFrame(results)
    # Add timing columns if they exist
    if 'train_time' in results_df.columns:
         results_df['train_time'] = results_df['train_time'].round(2)
    if 'predict_test_time' in results_df.columns:
         results_df['predict_test_time'] = results_df['predict_test_time'].round(2)
    if 'predict_bench_time' in results_df.columns:
         results_df['predict_bench_time'] = results_df['predict_bench_time'].round(2)

    results_csv_path = BASE_EXPERIMENT_DIR / "experiment_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved execution results to: {results_csv_path}")
else:
    print("\nExecution was skipped. Run the commands from the CSV file using a job scheduler or manually.")
    print("Example execution steps for one run (e.g., using bash):")
    print("  # Read parameters from experiment_commands.csv for a specific run_id")
    print("  RUN_ID='...'")
    print("  SPLIT_PKL_PATH=$(grep $RUN_ID experiment_commands.csv | cut -d',' -f11 | tail -n1)") # Adjust column index if needed
    print("  TARGET_SPLIT_PKL_PATH=$(grep $RUN_ID experiment_commands.csv | cut -d',' -f12 | tail -n1)")
    print("  TRAIN_CMD=$(grep $RUN_ID experiment_commands.csv | cut -d',' -f14 | tail -n1)")
    print("  PREDICT_TEST_CMD=$(grep $RUN_ID experiment_commands.csv | cut -d',' -f15 | tail -n1)")
    print("  # 1. Copy the correct split file")
    print("  cp \"$SPLIT_PKL_PATH\" \"$TARGET_SPLIT_PKL_PATH\"")
    print("  # 2. Run training")
    print("  $TRAIN_CMD")
    print("  # 3. Run prediction on test set")
    print("  $PREDICT_TEST_CMD")
    print("  # 4. (Manual) Parse results_json_path for Dice")
    print("  # 5. (Manual) Predict on benchmarking set and calculate Dice")
    print("  # 6. (Manual) Record time and Dice scores")

print("\nScript finished.")
