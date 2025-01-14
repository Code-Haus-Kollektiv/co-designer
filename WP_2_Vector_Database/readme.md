
# Co-Designer

This repository provides tools and scripts for creating and managing a Co-Designer environment for embedding generation, similarity search, and prediction for Rhino Grasshopper components. 

---

## Installation

Follow these steps to set up the development environment using Conda, a powerful Python environment manager.

### Prerequisites

Ensure you have Conda installed. If not, download and install it from [Conda's official site](https://docs.conda.io/en/latest/miniconda.html).

---

### Creating the Environment

To create the environment, use the `environment.yml` file, use:

```bash
conda env create -f environment.yml
```

### Exporting Environment
```bash
conda env export > environment.yml
```

## Setting Up VS Code Extensions

Run the following command to install the recommended extensions using Powershell:
```bash
Get-Content extensions.txt | ForEach-Object { code --install-extension $_ }
```