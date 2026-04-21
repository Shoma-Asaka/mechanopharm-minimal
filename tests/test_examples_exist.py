from pathlib import Path


def test_demo_script_exists():
    assert Path('examples/demo_minimal.py').exists()


def test_notebook_exists():
    assert Path('notebooks/reproduce_figures.ipynb').exists()
