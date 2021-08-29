# -*- coding: utf-8 -*-
"""
@author: 
@eamil: 
"""

from absl import app, flags
from mtd import test
from sys import argv, exit, stderr




if __name__ == "__main__":
    
    #### User configuration ####
    
    if len(argv) < 7 or len(argv) > 10:
        print('Use: {} [data_name] [attack] [p] [n]  [Q_max] [lamda] [batch_size=128] [epsilon=0.3] [batch_number=b1] '.format(argv[0]), file=stderr)
        exit(1)
    elif len(argv) >= 7:
        
        flags.DEFINE_string("data", argv[1], "used dataset.")
        flags.DEFINE_string("attack", argv[2], "evasion attack.")
        flags.DEFINE_integer("n", argv[4], "Number of student models.")
        flags.DEFINE_integer("p", argv[3], "Number of adv-trained student models.")
        flags.DEFINE_integer("Q_max", argv[5], "maximum number of queries for pool renewal.")
        flags.DEFINE_float("lamda", argv[6], "Noise scale.")
        flags.DEFINE_integer("test_set", 5000, "test set size.")
        flags.DEFINE_integer("class_nb", 10, "Number of labels.")
        if len(argv) == 7:
            flags.DEFINE_integer("batch", 128, "batch size.")
            flags.DEFINE_float("eps", 0.3, "Total epsilon for attacks.")
            flags.DEFINE_string("models_batch", 'b1', "the starting batch of student models.")
            
        elif len(argv) == 8:
            flags.DEFINE_integer("batch",argv[7] , "batch size.")
            flags.DEFINE_float("eps", 0.3, "Total epsilon for attacks.")
            flags.DEFINE_string("models_batch", 'b1', "the starting batch of student models.")
            
        elif len(argv) == 9:
            flags.DEFINE_integer("batch",argv[7] , "batch size.")
            flags.DEFINE_float("eps", argv[8], "Total epsilon for attacks.")
            flags.DEFINE_string("models_batch", 'b1', "the starting batch of student models.")
            
        elif len(argv) == 10:
            flags.DEFINE_integer("batch",argv[7] , "batch size.")
            flags.DEFINE_float("eps", argv[8], "Total epsilon for attacks.")
            flags.DEFINE_string("models_batch", argv[9], "the starting batch of student models.")
            

    
    app.run(test)
