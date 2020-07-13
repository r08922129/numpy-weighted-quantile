from  itertools import permutations
from weighted_quantile import weighted_quantile
import pickle
import numpy as np

from  itertools import permutations
import pickle

def add_sample(a, q, w, axis, test_sample,out=None,overwrite_input=False,keepdims=False):
    interpolation_list = ['lower','higher','midpoint','nearest','linear']
    for interpolation in interpolation_list:
        d ={'a':a,
            'q':q,
            'w':w,
            'axis':axis,
            'out':out,
            'overwrite_input':overwrite_input,
            'interpolation':interpolation,
            'keepdims':keepdims}
        test_sample.append(d)
def check_equal(param_list,error_samples):
    f = True
    for param_dict in param_list:
        a = param_dict['a']
        q = param_dict['q']
        w = param_dict['w']
        axis = param_dict['axis']
        out = param_dict['out']
        overwrite_input = param_dict['overwrite_input']
        interpolation = param_dict['interpolation']
        keepdims = param_dict['keepdims']
        
        result_a = quantile(a, q, w, axis=axis, out=out, overwrite_input=overwrite_input, interpolation=interpolation, keepdims=keepdims)
        result_b = quantile(a, q, axis=axis, out=out, overwrite_input=overwrite_input, interpolation=interpolation, keepdims=keepdims)
        if not np.allclose(result_a,result_b,equal_nan=True):
            error_samples.append(param_dict)
            print("Error occurs!")
            print("result_a",result_a)
            print("result_b",result_b)
            f = False
    if f:
        print("Pass!")


if __name__=="__main__":
    with open("test_sample.pkl",'rb') as f:
        test_sample = pickle.load(f)
    error_samples = []
    check_equal(test_sample,error_samples)
 
