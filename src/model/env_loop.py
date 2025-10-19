import csv
from pathlib import Path
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.env import BuildingEnv
from src.model.base_controller import BaseController

@dataclass
class MonitoringMsg:
    episode_len: int = 0
    episode_return: float = 0.0
    env_metrics: Optional[Dict[str, float]] = None
    trajectory: Optional[Dict[str, list]] = None


def run_eval_loop(
        env:BuildingEnv,
        controller:BaseController,
        heat_up_steps:int = 0
) -> MonitoringMsg:

    # Variables for monitoring
    monitoring = defaultdict(list)
    first_ts = 0
    episode_return = 0
    episode_price = 0
    episode_consumption = 0
    episode_len = 0

    # Reset the environment
    steps = 0
    done = False
    obs, info = env.reset()
    while not done:

        # Execute environment transaction
        action = controller.step(obs, info)
        obs, rew, terminal, terminated, info = env.step(action)
        done = terminal or terminated

        # Monitoring
        if steps >= heat_up_steps:
            # TODO: Not considering array-like actions
            if steps == heat_up_steps:
                first_ts = info["time"]
            monitoring["actions"].append(action)
            monitoring["rewards"].append(rew)
            state = info["state"]
            for k, v in state.items():
                monitoring[k].append(v)
            episode_return += rew
            # TODO: Env dependent variables (not generic)
            episode_price += state["price"]
            episode_consumption += state["e_bat"] + state["load"] - state["pv_gen"]
            episode_len += 1
        steps += 1

    return MonitoringMsg(
        episode_len=episode_len,
        episode_return=episode_return,
        env_metrics={
            "first_ts": first_ts,
            "total_price": episode_price,
            "total_consumption": episode_consumption
        },
        trajectory=monitoring
    )


def evaluate_and_report(
        log_dir: Path,
        n_runs: int,
        envs:List[BuildingEnv],
        controller:BaseController,
        heat_up_steps:int = 0,
        plot_trajectories:bool = False
) -> None:
    eval_metrics = defaultdict(list)
    eval_csv, trajectory_dir = _create_dir_structure(log_dir)
    
    # Create plots directory if plotting is enabled
    if plot_trajectories:
        plots_dir = log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

    for run in range(n_runs):
        for env_idx, env in enumerate(envs):
            env_name = f"eval_run_{run}_of_env_{env_idx}"
            msg = run_eval_loop(env, controller, heat_up_steps)
            write_evaluation_results(env_idx, env_name, eval_csv, msg, run)
            create_trajectory_csv(env_name, msg, trajectory_dir)
            
            # Plot trajectory if enabled
            if plot_trajectories and msg.trajectory:
                plot_path = plots_dir / f"{env_name}.png"
                plot_trajectory(msg.trajectory, plot_path)
                
            eval_metrics["episode_len"].append(msg.episode_len)
            eval_metrics["episode_return"].append(msg.episode_return)
            for k, v in msg.env_metrics.items():
                eval_metrics[k].append(v)

    for k, v in eval_metrics.items():
        print(f"{k}: {np.mean(v)} +- {np.std(v)}")


def create_trajectory_csv(
        env_name:str,
        msg:MonitoringMsg,
        trajectory_dir:Path
) -> None:
    fieldnames = msg.trajectory.keys()
    trajectory_csv = trajectory_dir / f"{env_name}.csv"
    with open(trajectory_csv, 'a', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(zip(*[msg.trajectory[key] for key in fieldnames]))


def write_evaluation_results(
        env_idx:int,
        env_name:str,
        eval_csv:Path,
        msg:MonitoringMsg,
        run:int
) -> None:
    result = {
        "name": env_name,
        "run": run,
        "env_idx": env_idx,
        "return": msg.episode_return,
        "env_steps": msg.episode_len}
    for k, v in msg.env_metrics.items():
        result[k] = v
    fieldnames = result.keys()
    file_exists = eval_csv.exists()
    with open(eval_csv, 'a', encoding='UTF8', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerow(result)


def _create_dir_structure(log_dir:Path) -> Union[Path, Path]:
    eval_dir = log_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    eval_csv = eval_dir / "evaluation_results.csv"
    trajectories_dir = eval_dir / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)
    return eval_csv, trajectories_dir

def plot_trajectory(trajectory_data: Dict[str, list], output_path: Path) -> None:
    """
    Plot trajectory data with multiple y-axes and save to file.
    
    Args:
        trajectory_data (Dict[str, list]): Dictionary containing trajectory data
        output_path (Path): Path to save the plot
    """
    # Convert trajectory data to dataframe for easier plotting
    df = pd.DataFrame(trajectory_data)
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Primary y-axis for load and pv
    if "load" in df.columns and "pv_gen" in df.columns:
        ax1.plot(df["load"], label="Load", color="blue")
        ax1.plot(df["pv_gen"], label="PV Generation", color="green")
        ax1.set_ylabel("Load and PV Generation", color="black")
    
    # Second y-axis for SOC
    if "soc" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df["soc"], label="SOC", color="red")
        ax2.set_ylabel("State of Charge (SOC)", color="red")
    else:
        ax2 = None
    
    # Third y-axis for price
    if "price" in df.columns:
        ax3 = ax1.twinx()
        # Offset the right spine for ax3
        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.spines["right"].set_visible(True)
        ax3.plot(df["price"], label="Price", color="orange")
        ax3.set_ylabel("Price", color="orange")
    else:
        ax3 = None
    
    # Add actions if available
    if "actions" in df.columns:
        action_color = "purple"
        if ax2 is None:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Actions", color=action_color)
        ax2.plot(df["actions"], label="Actions", color=action_color, linestyle="--")
    
    ax1.set_xlabel("Steps")
    ax1.set_title("Trajectory Data")
    
    # Combine legends from all axes
    lines = ax1.get_lines()
    labels = [line.get_label() for line in lines]
    
    if ax2 is not None:
        lines += ax2.get_lines()
        labels += [line.get_label() for line in ax2.get_lines()]
    
    if ax3 is not None:
        lines += ax3.get_lines()
        labels += [line.get_label() for line in ax3.get_lines()]
    
    ax1.legend(lines, labels, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)