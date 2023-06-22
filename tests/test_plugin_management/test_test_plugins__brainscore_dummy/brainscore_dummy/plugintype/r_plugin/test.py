from rpy2.robjects.packages import importr


def test_r():
    importr('base')
