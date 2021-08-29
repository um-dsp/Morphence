# -*- coding: utf-8 -*-
"""
@author: 
@eamil:
"""
from absl import app, flags
from mtd import generate_students
from sys import argv, exit, stderr

if __name__ == "__main__":
    
    #### User configuration ####
    
    if len(argv) < 6 or len(argv) > 9:
        print('Use: {} [data_name] [batch_number] [p] [n] [lambda] [batch_size=128] [epsilon=0.3] [max_iter=50]'.format(argv[0]), file=stderr)
        exit(1)
    elif len(argv) >= 6:
    
        flags.DEFINE_string("data", argv[1], "used dataset.")
        flags.DEFINE_string("models_batch", argv[2], "the starting batch of student models.")
        flags.DEFINE_integer("n", argv[4], "Number of student models.")
        flags.DEFINE_integer("p", argv[3], "Number of adv-trained student models.")
        flags.DEFINE_float("lamda", argv[5], "Noise scale.")
        flags.DEFINE_integer("test_set", 5000, "test set size.")
        if len(argv) == 6:
            flags.DEFINE_integer("batch", 128, "batch size.")
            flags.DEFINE_float("eps", 0.3, "Total epsilon for attacks.")
            flags.DEFINE_integer("max_iter", 50, "maximum number of iteration to reach max robustness.")
        elif len(argv) == 7:
            flags.DEFINE_integer("batch",argv[6] , "batch size.")
            flags.DEFINE_float("eps", 0.3, "Total epsilon for attacks.")
            flags.DEFINE_integer("max_iter", 50, "maximum number of iteration to reach max robustness.")
        elif len(argv) == 8:
            flags.DEFINE_integer("batch",argv[6] , "batch size.")
            flags.DEFINE_float("eps", argv[7], "Total epsilon for attacks.")
            flags.DEFINE_integer("max_iter", 50, "maximum number of iteration to reach max robustness.")
        elif len(argv) == 9:
            flags.DEFINE_integer("batch",argv[6] , "batch size.")
            flags.DEFINE_float("eps", argv[7], "Total epsilon for attacks.")
            flags.DEFINE_integer("max_iter", argv[8], "maximum number of iteration to reach max robustness.")
            
    
    
    
    app.run(generate_students)