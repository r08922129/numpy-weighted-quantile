from  itertools import permutations
from weighted_quantile import weighted_quantile
import pickle
import numpy as np

def add_sample(a,test_sample,out=None,overwrite_input=False,keepdims=False):
    w = np.ones_like(a)
    interpolation_list = ['lower','higher','midpoint','nearest','linear']
    q_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    axis_list = [None]
    for d in permutations(tuple(range(a.ndim))):
        axis_list.append(d)
    for interpolation in interpolation_list:
        for q in q_list:
            for axis in axis_list:
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
        
        result_a = weighted_quantile(a, q, w, axis=axis, out=out, overwrite_input=overwrite_input, interpolation=interpolation, keepdims=keepdims)
        result_b = np.quantile(a, q, axis=axis, out=out, overwrite_input=overwrite_input, interpolation=interpolation, keepdims=keepdims)
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
 
