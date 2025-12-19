# Isaac Sim Navigation Demo

This demo allows you to test the navigation performance for 10 episodes each of `fine_grained` and `coarse_grained` tasks.

## Prerequisites

Before running the demo, please follow the [this guide](https://internrobotics.github.io/user_guide/internnav/quick_start/simulation.html) to:
1. Download the Isaac Simulator.
2. Create and set up a conda environment for the simulation.

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
