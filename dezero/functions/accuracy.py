from dezero.core import Variable, as_array, as_variable


def accuracy(y: Variable, t: Variable) -> Variable:
    y, t = as_variable(y), as_variable(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = pred == t.data
    acc = result.mean()
    return Variable(as_array(acc))
