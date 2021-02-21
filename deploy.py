import os
import libtmux

class Deployer:

    def __init__(self, session_name='metric-learning-exp'):
        self.sname = session_name
        self.session = libtmux.Server().new_session(self.sname, kill_session=True)

    def __call__(self, fname, workers):
        fname = os.path.abspath(fname)
        cmd = f"python3 src/manager.py {fname} {workers}"
        pane =  self.session.attached_window.attached_pane
        pane.send_keys('source venv/bin/activate')
        pane.send_keys(cmd)

if __name__ == '__main__':
    import sys
    d = Deployer()
    d(sys.argv[1], sys.argv[2])

        