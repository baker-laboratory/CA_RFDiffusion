import traceback
import os
from inspect import signature
import pickle
import datetime
import rf2aa.tensor_util

from icecream import ic

LOG_ONLY_KEY = 'log_only_key'

class Counter:
    def __init__(self):
        self.i=0
        self.last_pickle = None

def pickle_function_call_wrapper(func, output_dir='pickled_inputs', include_outputs=True, minifier=lambda x: None):
    counter = Counter()
    os.makedirs(output_dir, exist_ok=True)
            # pickle.dump({'args': args, 'kwargs': kwargs}, fh)
    def wrapper(*args, **kwargs):
        """
        Wrap the original function call to print the arguments before
        calling the intended function
        """
        log_only = kwargs.pop(LOG_ONLY_KEY, {})
        assert isinstance(log_only, dict)

        counter.i += 1
        func_sig = signature(func)
        # Create the argument binding so we can determine what
        # parameters are given what values
        argument_binding = func_sig.bind(*args, **kwargs)
        argument_map = argument_binding.arguments
        assert set(log_only.keys()).isdisjoint(set(argument_map.keys())), f'{log_only.keys()}, {argument_map.keys()}'
        argument_map.update(log_only)

        # Perform the print so that it shows the function name
        # and arguments as a dictionary
        path = os.path.join(output_dir, f'{counter.i:05d}.pkl')
        print(f"logging {func.__name__} arguments: {[k for k in argument_map]} to {path}")
        argument_map['stack'] = traceback.format_stack()
        
        # for k, v in argument_map.items():
        #     if hasattr(v, 'detach'):
        #         argument_map[k] = v.cpu().detach()
        
        raw_o = func(*args, **kwargs)

        if include_outputs:
            # o = []
            # for i, v in enumerate(raw_o):
            #     if hasattr(v, 'detach'):
            #         o.append(v.cpu().detach())
            #     else:
            #         o.append(v)
            o = {f'out_{i}': v for i,v in enumerate(raw_o)}
            argument_map.update(o)
        
        minifier(argument_map)
        argument_map = rf2aa.tensor_util.cpu(argument_map)

        # for k, v in argument_map.items():
        #     try:
        #         print(f'{k}: {v.device}')
        #     except Exception as e:
        #         print(f'{k}: {type(v)} has no device: {v}')
        with open(path, 'wb') as fh:
            pickle.dump(argument_map, fh)
        counter.last_pickle = path

        return raw_o

    return wrapper, counter

def wrap_it(wrapper, instance, method, **kwargs):
    class_method = getattr(instance, method)
    wrapped_method, extra = wrapper(class_method, **kwargs)
    setattr(instance, method, wrapped_method)
    return extra



def pickle_function_call(instance, method, subdir, include_outputs=True, minifier=lambda x:None):
	output_dir = os.path.join(os.getcwd(), 'pickled_inputs', subdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
	extra = wrap_it(pickle_function_call_wrapper, instance, method, output_dir=output_dir, include_outputs=include_outputs, minifier=minifier)
	return output_dir, extra

# For testing
if __name__=='__main__':
	import glob
	class Dog:
		def __init__(self, name):
			self.name = name
		def bark(self, arg, kwarg=None):
			print(f'{self.name}:{arg}:{kwarg}')

	dog = Dog('fido')
	dog.bark('ruff')

	output_dir, extra = pickle_function_call(dog, 'bark', 'debugging')

	dog.bark('ruff', kwarg='wooof')
	ic(extra.i, extra.last_pickle)

	for p in glob.glob(os.path.join(output_dir, '*')):
		print(p)
		with open(p, 'rb') as fh:
			print(pickle.load(fh))
