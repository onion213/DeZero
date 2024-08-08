import os
import subprocess
import tempfile

from dezero.core import Function, Variable


def _dot_var(v: Variable, verbose: bool = False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f: Function):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output: Variable, verbose: bool = True):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f: Function):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output: Variable, verbose: bool = True, to_file: str = "graph.png"):
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    dot_graph = get_dot_graph(output, verbose)

    # Save the dot graph to a file
    with open(temp_file.name, "w") as f:
        f.write(dot_graph)

    # Convert the dot graph to an image
    extension = os.path.splitext(to_file)[1][1:]
    cmd = f"dot {temp_file.name} -T {extension} -o {to_file}"
    subprocess.run(cmd, shell=True)
