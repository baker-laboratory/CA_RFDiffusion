# This file exists so that we don't load pyrosetta by default when people don't need it
# It takes several seconds to load, displays a message, and might not be present in their environment

# You'll probably import it like this:
# from rf_diffusion.import_pyrosetta import pyrosetta as pyro
# from rf_diffusion.import_pyrosetta import rosetta as ros

# Then use it like this:
# pyro().pose_from_file()
# ros().core.pose.Pose()

pyrosetta_flags = None
def prepare_pyrosetta(conf):
    '''
    Call this right after main() such that pyrosetta can later be loaded without conf
    '''
    global pyrosetta_flags
    if 'pyrosetta_flags' in conf:
        pyrosetta_flags = conf.pyrosetta_flags
    else:
        pyrosetta_flags = ''

_pyrosetta = None
def pyrosetta():
    '''
    Call this to get access to pyrosetta
    '''
    global _pyrosetta
    if _pyrosetta is None:
        assert pyrosetta_flags is not None, 'prepare_pyrosetta() was never called!'

        import pyrosetta
        assert not pyrosetta.rosetta.basic.was_init_called(), 'Please do not import pyrosetta yourself. Use rf_diffusion.import_pyrosetta.pyrosetta()'
        pyrosetta.init(pyrosetta_flags, silent=True)

        _pyrosetta = pyrosetta
    return _pyrosetta

def rosetta():
    '''
    Call this to get access to pyrosetta.rosetta
    '''
    return pyrosetta().rosetta

