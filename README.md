# Iris Classification Agent

This project implements a classification agent for the classic **Iris dataset**. It follows a clear structure compatible with evaluation platforms such as `terminal-bench`, and includes:

- an **agent** that trains and generates predictions, and  
- a **grader** that evaluates the model's predictions and returns a score.
  
## 📂 Project Structure
- `agent.py` → The classifier script (generates predictions)
- `grader.py` → The evaluation script (computes accuracy)
- `data/iris.csv` → The dataset used for training/testing
- `workdir/sol.csv` → Predictions saved by the agent
- `requirements.txt` → Python dependencies
- `Dockerfile` → Docker configuration to run the agent
- `task.yaml` → Task config for terminal-bench

## 🚀 How to Run

### 1. Run the Agent

The agent trains on 80% of the Iris dataset and predicts the remaining 20%. Predictions are saved to `workdir/sol.csv`.

```bash
python3 agent.py
```
### 2. Run the Grader
The grader compares predictions (workdir/sol.csv) with ground truth labels in data/iris.csv and returns a JSON result:

```bash
python3 grader.py
```
Example output:
{
  "score": 0.9667,
  "feedback": "Accuracy: 0.9667"
}
### 🐳 Run with Docker
You can run the agent in a container using Docker:

```bash
docker build -t iris-agent .
docker run --rm -v $(pwd)/data:/workdir/data -v $(pwd)/workdir:/workdir iris-agent
```
This mounts your local data and workdir folders into the container so the model can access inputs and write predictions.

📦 Dependencies
Install dependencies using:

```bash
pip install -r requirements.txt
```
requirements.txt:

```bash
pandas
scikit-learn
```
📝 task.yaml (for terminal-bench)
This defines the task structure for benchmarking:

```bash
name: iris-classification
description: Iris classification agent using scikit-learn
agent_file: agent.py
grader_file: grader.py
command: python3 agent.py
```
🧪 Example (with terminal-bench)
To test the model in a benchmarking environment (after proper setup):

```bash
terminal-bench run --dataset ./tests --model .
```
Make sure the dataset and model follow the required directory structure for terminal-bench.

🧠 Notes:

- The agent must write predictions to workdir/sol.csv

- The grader expects data/iris.csv and the prediction file to exist

- You can evaluate locally using only Python, or inside Docker

