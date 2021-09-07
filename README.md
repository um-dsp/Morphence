# Morphence: Moving Target Defense Against Adversarial Examples
This repository contains the source code accompanying our ACSAC'21 paper [Morphence: Moving Target Defense Against Adversarial Examples]( https://arxiv.org/abs/2108.13952). 

The following detailed instructions can be used to reproduce our results in Section 4.2 (Table 1). The user is also able to try parameters configuration other than what is adopted in the paper.

A GPU hardware environment is mandatory to run the code.

### Installation
It is required to create a separate python3 environment. Then use the following commands:

```$ git clone https://github.com/um-dsp/Morphence.git ```

```$ cd Morphence ```

```$ pip install -r requirements.txt ```

### Morphence Pool Generation
You can generate the pool of models from scratch **A** or you can download previously generated student models **B**.

**A: Generate from scratch**

*Note*: Generating and retraining adversarially-trained models could take several hours. For a quicker experiment you can skip to step **B**.
```
$ python generate_students.py [data_name] [batch_number] [p] [n] [lambda] [batch_size=128] [epsilon=0.3] [max_iter=50]
```
MNIST example:  ``` $ python generate_students.py MNIST b1 5 10 0.1 ```

CIFAR10 example:  ``` $ python generate_students.py CIFAR10 b1 9 10 0.05 ```

In order to generate 5 batches (pools of models) we execute the same command for b2, b3, b4 and b5.

**B: Use previously generated models**

(Skip to Morphence Evaluation if you already used step **A**)

First create a folder called "experiments" (i.e ```/Morphence/experiments ```.
Next, run the following commands to download the student models:
```
$ cd experiments
```
For MNIST: 

```$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1im49tMXgMHWapvA5UXmfhEQw7WnfRFzR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1im49tMXgMHWapvA5UXmfhEQw7WnfRFzR" -O MNIST.zip && rm -rf /tmp/cookies.tx```

For CIFAR 

```$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1im49tMXgMHWapvA5UXmfhEQw7WnfRFzR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1im49tMXgMHWapvA5UXmfhEQw7WnfRFzR" -O CIFAR.zip && rm -rf /tmp/cookies.tx```

Finally, unzip the models: ```unzip [data_name].zip -d [data_name]```

### Morphence Evaluation
```
$ python test.py [data_name] [attack] [p] [n] [Q_max] [lamda] [batch_size=128] [epsilon=0.3] [batch_number=b1]
```
**Note**: In case **Step B** is used for model generation, you have to use the default configuration for the evaluation. Otherwise, you have to use the same configuration adopted in **Step A**.

```[attack]``` can be: CW, FGS, SPSA or NoAttack.
example:  ``` $ python test.py CIFAR10 CW 9 10 1000 0.05 ```

### Fixed Baseline models Evaluation

Undefended : ```$ python test_base.py [data_name] [attack] [batch_size=128] [epsilon=0.3] ```

Adversarially-trained : ```$ python test_adv.py [data_name] [attack] [batch_size=128] [epsilon=0.3] ```

### Copycat Models

The used Copycat code is a modified version of the original code provided in : https://github.com/jeiks/Stealing_DL_Models/tree/master/Framework

`$ cd Copycat/Framework`

##### Steal base model:
For CIFAR10:```$ python copycat/steal_train.py copycat_base_CNN_CIFAR10.pth [path-to-base-model] CIFAR10```

For MNIST:```$ python copycat/steal_train.py copycat_base_CNN_MNIST.pth [path-to-base-model] MNIST```

##### Steal adversarially trained model:
For CIFAR10:```$ python copycat/steal_train.py copycat_base_adv_CNN_CIFAR10.pth [path-to-defended-model] CIFAR10```

For MNIST:```$ python copycat/steal_train.py copycat_base_adv_CNN_MNIST.pth [path-to-defended-model] MNIST```

##### Steal Morphence:
For CIFAR10:```$ python copycat/steal_train.py copycat_mtd_CNN_CIFAR10.pth Morphence CIFAR10```

For MNIST:```$ python copycat/steal_train.py copycat_mtd_CNN_MNIST.pth Morphence MNIST```

##### Test model extraction of fixed models:

For CIFAR10:```$ python copycat/test.py [path-to-copycat-model] [path-to-target-model] CIFAR10```

For MNIST:```$ python copycat/test.py [path-to-copycat-model] [path-to-target-model] MNIST```

[path-to-copycat-model] can be either the path to the copycat model of the base model or the adversarially trained model.

[path-to-target-model] can be either the path to the base model or the adversarially trained model.

##### Test model extraction of Morphence:
For CIFAR10:```$ python copycat/morph_test.py copycat_mtd_CNN_CIFAR10.pth CIFAR10```

For MNIST:```$ python copycat/morph_test.py copycat_mtd_CNN_MNIST.pth MNIST```
