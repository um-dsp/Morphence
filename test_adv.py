# -*- coding: utf-8 -*-
"""

@author: Abderrahmen Amich
@email:  aamich@umich.edu
"""

from absl import app, flags
from mtd import test_adv
from sys import argv, exit, stderr




if __name__ == "__main__":
    
    #### User configuration ####
    
    if len(argv) < 3 or len(argv) > 5:
        print('Use: {} [data_name] [attack] [batch_size=128] [epsilon=0.3] '.format(argv[0]), file=stderr)
        exit(1)
    elif len(argv) >= 3:
        
        flags.DEFINE_string("data", argv[1], "used dataset.")
        flags.DEFINE_string("attack", argv[2], "evasion attack.")
        flags.DEFINE_integer("test_set", 5000, "test set size.")
        flags.DEFINE_integer("class_nb", 10, "Number of labels.")
        if len(argv) == 3:
            flags.DEFINE_integer("batch", 128, "batch size.")
            flags.DEFINE_float("eps", 0.3, "Total epsilon for attacks.")
            
            
        elif len(argv) == 4:
            flags.DEFINE_integer("batch",argv[3] , "batch size.")
            flags.DEFINE_float("eps", 0.3, "Total epsilon for attacks.")
            
            
        elif len(argv) == 5:
            flags.DEFINE_integer("batch",argv[3] , "batch size.")
            flags.DEFINE_float("eps", argv[4], "Total epsilon for attacks.")
        
            

    
    app.run(test_adv)
