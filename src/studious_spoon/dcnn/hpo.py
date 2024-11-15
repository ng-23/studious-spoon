from marshmallow import Schema
from sklearn.model_selection import ParameterGrid

registered_searchers = {}

def register_searcher(name):
    def wrapper(cls):
        registered_searchers[name] = cls
        return cls
    return wrapper

@register_searcher('grid')
class GridSearcher():
    def __init__(self, search_spaces:dict[str,list], sample_schema:Schema|None=None):
        self._sample_schema = sample_schema
        self._param_grid = self._make_param_grid(search_spaces)
        self._len = len(self._param_grid)
        self._curr_sample = 0

    def _make_param_grid(self, search_spaces:dict[str,list]):
        if len(search_spaces) == 0:
            raise Exception('Search space cannot be empty')
        
        param_grid = {}

        # maps an unpacked param name (search space name + separator + param name) to a tuple of the search space and original param name
        unpacked_param_names = {}

        # unpack each search space dict
        for search_space_name in search_spaces:
            search_space = search_spaces[search_space_name]
            for param in search_space:
                param_name = f'{search_space_name}_{param}'
                param_grid[param_name] = search_space[param]
                unpacked_param_names[param_name] = (search_space_name, param)
        
        # generate all possible combinations of params (samples)
        grid = list(ParameterGrid(param_grid))

        # prepare new grid, where each sample has its params repacked into their associated dicts
        refined_grid = [{search_space_name:{} for search_space_name in search_spaces} for _ in range(len(grid))]

        # repack params back into their associated dicts
        # and optionally verify each sample conforms to a certain schema
        for i in range(len(grid)):
            sample = grid[i]
            for param in sample:
                search_space_name, param_name = unpacked_param_names[param]
                refined_grid[i][search_space_name][param_name] = sample[param]
            if self._sample_schema is not None:
                refined_grid[i] = self._sample_schema.load(refined_grid[i])

        return refined_grid
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._curr_sample >= self._len:
            raise StopIteration
        
        sample = self._param_grid[self._curr_sample]
        self._curr_sample += 1
        return sample
    
    def __len__(self):
        return len(self._param_grid)
    