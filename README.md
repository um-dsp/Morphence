# Morphence: Moving Target Defense Against Adversarial Examples
This repository contains the source code accompanying our ACSAC'21 paper [Morphence: Moving Target Defense Against Adversarial Examples]( https://arxiv.org/abs/2108.13952) and its extension to [Morphence-2.0: Evasion-Resilient Moving Target Defense Powered by Out-of-Distribution Detection](https://arxiv.org/abs/2206.07321).

The following detailed instructions can be used to reproduce our results in Section 4.2 (Table 1). The user is also able to try parameters configuration other than what is adopted in the paper. A GPU hardware environment is mandatory to run the code.
***
### Morphence Demo on MNIST Dataset
This demo on the MNIST dataset gives you a flavor of Morphence in action:

[![Morphence: MNIST demo](http://i3.ytimg.com/vi/8hkp_U0iY4o/maxresdefault.jpg)](https://youtu.be/8hkp_U0iY4o)
***



### Step-1: Downloading Morphence and Installing Dependencies 
It is first required to create a separate python3 environment. Then execute the following commands from within the newly created python3 environment:

```$ git clone https://github.com/um-dsp/Morphence.git ```

```$ cd Morphence ```

```$ pip install -r requirements.txt ```
***
### Step-2: Morphence Pool Generation
You can generate the pool of models either by downloading previously generated student models with **Option A** (faster) or generate pool of models from scratch with **Option B** (slower).

**Option A: Use previously generated models (faster)**

First create a folder called "experiments" (i.e ```/Morphence/experiments ```).
Next, run the following commands to download the student models:
```
$ cd experiments
```
For MNIST: 

```$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1im49tMXgMHWapvA5UXmfhEQw7WnfRFzR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1im49tMXgMHWapvA5UXmfhEQw7WnfRFzR" -O MNIST.zip && rm -rf /tmp/cookies.tx```

For CIFAR: 

```$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WC91-yvPznjtZU503ehH7XG5wohzNtTO' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WC91-yvPznjtZU503ehH7XG5wohzNtTO" -O CIFAR10.zip && rm -rf /tmp/cookies.tx```

For Windows users follow this direct download [link](https://drive.google.com/drive/folders/1Ohdc9BXVLq883ZCz8O5WeFzydnaUok8S?usp=sharing)

Finally, unzip the models: ```unzip [data_name].zip```

To experiment Morphence2.0 on CIFAR10, downolad [Cifar10](https://drive.google.com/file/d/17WZP5ZWs0qwbnu0K96leas1R97b3RMbK/view?usp=sharing) SSD model and place it in ```SSD/models```/.

**Option B: Generate from scratch (may take a while)**

*Note*: Generating and retraining adversarially-trained models could take several hours. 
```
$ python generate_students.py [data_name] [batch_number] [p] [n] [lambda] [batch_size=128] [epsilon=0.3] [max_iter=50]
```
MNIST example:  ``` $ python generate_students.py MNIST b1 5 10 0.1 ```

CIFAR10 example:  ``` $ python generate_students.py CIFAR10 b1 9 10 0.05 ```

In order to generate 5 batches (pools of models) we execute the same command for b2, b3, b4 and b5.


***
### Step-3: Morphence Evaluation

The following command initiates a Morphence framework and performs [attack].

You can try both Morphence versions by configuring the version parameter to 1 or 2

```
$ python test.py [data_name] [attack] [p] [n]  [Q_max] [lamda] [version] [batch_size=100] [epsilon=0.3] [ssd_training_mode=SIMCLR] [arch =resnet50] [ckpt=./models_ssd]  [data_dir = data directory path] [data-mode = for SSD {base,ssl,org}] [normalize = normalise data for SSD] [batch_number=b1]  [size = ssd_getDatasets resize param ]

```
**Note**: If **Option A** is used for model generation, you have to use the default configuration for the evaluation. Otherwise, you have to use the same configuration adopted in **Option B**.

```[attack]``` can be: CW, FGS, SPSA or NoAttack.

example:

MNIST: ```$ python test.py MNIST FGS  5 10 1000 0.1 2 200 0.3 SimCLR resnet18 SSD/models/MNIST.pth.tar  ./data/ base false  b1 28```

CIFAR10: ``` $ python test.py CIFAR10 FGS  9 10 1000 0.05 2 200 0.3 SimCLR resnet50 SSD/models/CIFAR10.pth  ./data/ base false  b1 32 ```

### Fixed Baseline models Evaluation

* Undefended model : ```$ python test_base.py [data_name] [attack] [batch_size=128] [epsilon=0.3] ```

* Adversarially-trained model : ```$ python test_adv.py [data_name] [attack] [batch_size=128] [epsilon=0.3] ```

### [Extra] Model extraction evaluation

Note: The following experiments do not take part in the paper.

The used Copycat code is a modified version of the original code provided in : https://github.com/jeiks/Stealing_DL_Models/tree/master/Framework

`$ cd Copycat/Framework`

##### Steal base model:
* For CIFAR10:```$ python copycat/steal_train.py copycat_base_CNN_CIFAR10.pth [path-to-base-model] CIFAR10```

* For MNIST:```$ python copycat/steal_train.py copycat_base_CNN_MNIST.pth [path-to-base-model] MNIST```

##### Steal adversarially trained model:
* For CIFAR10:```$ python copycat/steal_train.py copycat_base_adv_CNN_CIFAR10.pth [path-to-defended-model] CIFAR10```

* For MNIST:```$ python copycat/steal_train.py copycat_base_adv_CNN_MNIST.pth [path-to-defended-model] MNIST```

##### Steal Morphence:
* For CIFAR10:```$ python copycat/steal_train.py copycat_mtd_CNN_CIFAR10.pth Morphence CIFAR10```

* For MNIST:```$ python copycat/steal_train.py copycat_mtd_CNN_MNIST.pth Morphence MNIST```

##### Test model extraction of fixed models:

* For CIFAR10:```$ python copycat/test.py [path-to-copycat-model] [path-to-target-model] CIFAR10```

* For MNIST:```$ python copycat/test.py [path-to-copycat-model] [path-to-target-model] MNIST```

  [path-to-copycat-model] can be either the path to the copycat model of the base model or the adversarially trained model.

 [path-to-target-model] can be either the path to the base model or the adversarially trained model.

##### Test model extraction of Morphence:
* For CIFAR10:```$ python copycat/morph_test.py copycat_mtd_CNN_CIFAR10.pth CIFAR10```

* For MNIST:```$ python copycat/morph_test.py copycat_mtd_CNN_MNIST.pth MNIST```
***
### How to Cite Morphence
If you use Morphence in a scientific publication, please cite the corresponding paper as follows:

```
Abderrahmen Amich and Birhanu Eshete. Morphence: Moving Target Defense Against Adversarial Examples. In Proceedings of the 37th Annual Computer Security Applications Conference, ACSAC, 2021.
```

BibTex entry:
```BibTex
inproceedings{Morphence21,
 author    = {Abderrahmen Amich and Birhanu Eshete},
  title     = {{Morphence: Moving Target Defense Against Adversarial Examples}},
booktitle = {Annual Computer Security Applications Conference (ACSAC'21), December 6--10, 2021, Virtual Event, USA},
publisher ={ACM},
pages = {61--75},
 year      = {2021}
  }
```
***
