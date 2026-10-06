from src.features import extract_metrics


def test_basic_metrics():
    code = '''\ndef f(x):\n    for i in range(x):\n        if i % 2 == 0:\n            print(i)\n'''
    m = extract_metrics(code)
    assert m.functions == 1
    assert m.loops == 1
    assert m.conditionals == 1
    assert m.complexity >= 3
    assert m.nesting >= 3
