# Get Started

We highly recommend installing
`zenpyre` in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to avoid dependency conflicts.

## Using uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package installer and resolver:

```shell
uv pip install zenpyre
```

**Install with all optional dependencies:**

```shell
uv pip install zenpyre[all]
```

**Install with specific optional dependencies:**

```shell
uv pip install zenpyre[numpy,torch]  # with NumPy and PyTorch
```

## Using pip

Alternatively, you can use `pip`:

```shell
pip install zenpyre
```

**Install with all optional dependencies:**

```shell
pip install zenpyre[all]
```

**Install with specific optional dependencies:**

```shell
pip install zenpyre[numpy,torch]  # with NumPy and PyTorch
```

## Installing from source

To install `zenpyre` from source, you can follow the steps below.

First, clone the git repository:

```shell
git clone git@github.com:durandtibo/zenpyre.git
cd zenpyre
```

**Note**: `zenpyre` requires Python 3.10 or higher.

It is recommended to create a virtual environment (this step is optional).
To create a virtual environment, you can use the following command:

```shell
make setup-venv
```

This command automatically creates a virtual environment using [`uv`](https://docs.astral.sh/uv/).
When the virtual environment is created, you can activate it with the following command:

```shell
source .venv/bin/activate
```

Then, you should install the required packages to use `zenpyre` with the following command:

```shell
inv install --docs-deps
```

This command will install all the required packages. You can also use this command to update the
required packages. This command will check if there is a more recent package available and will
install it. Finally, you can test the installation with the following command:

```shell
inv unit-test --cov
```
