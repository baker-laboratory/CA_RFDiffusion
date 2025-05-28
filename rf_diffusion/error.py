import contextlib


@contextlib.contextmanager
def context(msg, cleanup=lambda:None):
    try:
        yield
    except Exception as ex:
        msg = '{}: {}'.format(msg, ex.args[0]) if ex.args else str(msg)
        ex.args = (msg,) + ex.args[1:]
        cleanup()
        raise


if __name__ == '__main__':
    def do_something(val):
        return 1/val

    val = 0
    with context(f'processing {val}'):
        do_something(val)
