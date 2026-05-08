# Digit Classification Setup Guide

## Prerequisites

Download Miniconda from:

[https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)

Install Conda according to your operating system.

Verify the installation:

```bash
conda --version
```

---

## Open Anaconda Prompt

Search for **Anaconda Prompt** in the Windows Start Menu and open it.

Initialize Conda for PowerShell:

```bash
conda init powershell
```

---

## Create Conda Environment

Open VS Code and create a new terminal.

Create a new Conda environment:

```bash
conda create -n digit python=3.14
```

Activate the environment:

```bash
conda activate digit
```

---

## Create `requirements.txt`

Create the requirements file:

```bash
nano requirements.txt
```

Add the following packages inside `requirements.txt`:

```txt
matplotlib
scikit-learn
```

---

## Install Required Packages

Run:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Execute the script:

```bash
python plot_digits_classification.py
```
