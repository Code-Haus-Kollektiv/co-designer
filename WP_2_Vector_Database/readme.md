## Installation

Using Conda as the python environment managewed

```bash
conda create -n co-designer python=3.12
conda activate co-designer
```


### Packages
```bash
conda install conda-forge::chromadb
conda install conda-forge::colorama
conda install conda-forge::pyautogen
conda install conda-forge::sentence-transformers
```

### Exporting Environment
```bash
# Export
conda env export > environment.yml
# Create 
conda env create -f environment.yml

```