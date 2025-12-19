# Isaac Sim Navigation Demo

This demo allows you to test the navigation performance for 10 episodes each of `fine_grained` and `coarse_grained` tasks.

## Prerequisites

Before running the demo, please follow the [this guide](https://internrobotics.github.io/user_guide/internnav/quick_start/simulation.html) to:
1. Download the Isaac Simulator.

We use Isaac Sim 4.5.0. Download it from [offical page](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html), which is a zip file. Simply unzip it to a folder (Let's say ISAACSIM_ROOT.).

2. Create and set up a conda environment for the simulation.
```
conda create -n goodnav python=3.10
conda activate goodnav
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas
pip install scipy==1.10.1
cd ISAACSIM_ROOT
source setup_conda_env.sh
```

3. Tested environment

We test the demo on Ubuntu 24 with RTX4090 (Driver Version: 570.195.03     CUDA Version: 12.8 ).

## Setup

Download the necessary scene files and code:
   ```bash
   git clone https://github.com/william13077/IAmGoodNavigator
   cd IAmGoodNavigator
   bash download.sh
   ```


## How to Run

To run a specific demo episode, use the following command:
```bash
python demo.py --task <task_type> --index <episode_index>
```
*   `--task`: Choose either `fine` or `coarse`.
*   `--index`: Choose an index from `0` to `9`.

Example:
```bash
python demo.py --task fine --index 0
```

The script will load the specific scene USD file and data based on the provided task and index.

## Interaction & Controls

### Interface
*   **Instruction Panel:** Once the simulation starts, you will see an instruction panel on the left side of the screen displaying the task descriptions.
*   **Camera Setup:** To see the robot's perspective:
    1. Click the **Perspective** button at the top (slightly to the left of the center).
    2. Select **Cameras**.
    3. Select **FloatingCamera**.

### Controls
*   **W:** Move forward
*   **S:** Move backward
*   **A:** Turn left
*   **D:** Turn right

### Completion
Once you have reached the goal or completed the task, press the **Enter** key to view your performance evaluation.
