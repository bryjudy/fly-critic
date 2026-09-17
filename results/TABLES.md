# Aggregated results

Produced by the three aggregators on the full run set. Oracle for the odor tasks is 33.5 per episode, for ToolWorld 105.

## Odor tasks, head-to-heads, ToolWorld (flycritic.aggregate)

```
config                   seeds  iters        R/episode     pre    post  punish hits
b1_bandit_learned_kc_prior     3   1500     136.7 +/-12.9    73.4    63.3          0.0
b1_bandit_none               3   1500      99.5  +/-2.3    52.6    46.9          0.0
b1_bandit_none_big           2   1500      95.8  +/-3.9    49.7    46.1          0.0
b1_bandit_none_hi            3   1500      87.2  +/-4.8    47.0    40.1          0.0
cx2_mb_kc_ctx                2    400      57.1  +/-0.0    29.0    28.1          5.2
b1_darkroom_learned_kc_prior     3   1000      38.0  +/-4.9    38.0     0.0          0.0
b1_darkroom_mb_nofeat_kc     1   1000      38.0            38.0     0.0          0.0
b1_darkroom_none             3   1000      31.5  +/-4.3    31.5     0.0          0.0
ch_shuffled                  1    600      30.6            15.7    14.9          1.7
ch_mb                        1    600      30.4            15.6    14.8          1.7
ch_mb_kc                     3    400      30.0  +/-0.3    15.2    14.8          1.9
hh_hybrid_kc                 3   1200      28.9  +/-0.4    14.8    14.1          2.4
hh_learned_randkc_prior      3   1200      28.5  +/-0.2    14.2    14.3          3.0
hh_learned_kc_prior          3   1200      28.2  +/-0.4    14.0    14.2          3.1
tw15_learned_kc_prior        3    600      28.0  +/-3.2    12.5    15.5         73.6
hh_learned_kc                3   1200      26.8  +/-0.6    13.7    13.1          3.6
tw_mb_nofeat_kc              2    600      24.1  +/-0.2    11.1    13.0         76.7
tw_learned_kc_prior          2    600      23.0  +/-0.0    10.5    12.5         77.6
long_mb_nofeat_kc            3   1200      21.6  +/-5.3    13.4     8.2          4.4
b1_darkroom_none_hi          3   1000      20.2  +/-0.6    20.2     0.0          0.0
tw_none                      2    600      17.6  +/-0.1     8.7     8.9         81.9
tw15_none                    3    600      17.5  +/-0.1     8.7     8.8         81.9
tw_none_hi                   1    600      17.3             8.6     8.8         81.5
ch_mb_nofeat_kc              3    400      17.3  +/-0.1     8.8     8.5         11.7
ch_shuf_nofeat_kc            3    400      16.1  +/-0.8     8.3     7.8         12.8
ch_uniform_nofeat_kc         3    400      12.8  +/-0.3     8.2     4.6         10.9
cx2_mb_nofeat_kc_ctx         2    400      11.3  +/-1.0     8.8     2.5         29.7
ch_collapse_nofeat_kc        3    400       6.3  +/-0.2     7.2    -0.9         23.6
ch_scalar                    1    600       0.2             0.1     0.1         16.2
cx_mb_nofeat_kc_ctx          2    400       0.0  +/-0.0     0.0    -0.0         11.3
cx_mb_nofeat_kc              2    400       0.0  +/-0.0     0.0     0.0         17.6
cx_mb_kc                     2    400       0.0  +/-0.1     0.0     0.0         17.0
hh_none                      3   1200       0.0  +/-0.1     0.0     0.0         16.4
hh_learned_h                 3   1200       0.0  +/-0.1     0.0     0.0         16.1
cx_mb_kc_ctx                 2    400       0.0  +/-0.0     0.1    -0.0          9.3
cx_none                      2    400       0.0  +/-0.1     0.0     0.0         10.5
ch_none                      1    600      -0.1             0.0    -0.1         14.8
ch_mb_nofeat                 1    600      -0.1            -0.0    -0.1         14.4
cx2_mb_nofeat_kc             2    400      -0.1  +/-0.0    -0.0    -0.0         33.5
cx2_mb_kc                    2    400      -0.1  +/-0.0    -0.0    -0.1         33.0
cx2_none                     2    400      -0.1  +/-0.0    -0.1    -0.1         33.7
(oracle = 33.5/episode; final-100-iteration training averages; pre/post = reward before/after the reversal)
```

## POPGym budget pass, completed 3M-step runs only (flycritic.popgym.results --complete_only)

```
env                                    fly (ours)       gru (ours)  GRU (paper)         best (paper)
AutoencodeEasy                   -0.462+/-0.002n3 -0.454+/-0.011n3       -0.283         -0.283 (GRU)
AutoencodeMedium                 -0.467+/-0.006n3 -0.470+/-0.005n2       -0.425      -0.420 (IndRNN)
AutoencodeHard                   -0.473+/-0.004n3 -0.477+/-0.003n3       -0.456      -0.448 (IndRNN)
CountRecallEasy                  -0.667+/-0.017n3 -0.655+/-0.008n3        0.177         0.509 (LSTM)
CountRecallMedium                -0.889+/-0.028n3 -0.763+/-0.004n3       -0.528      -0.519 (PosMLP)
CountRecallHard                  -0.872+/-0.001n3 -0.872+/-0.001n3       -0.475  -0.470 (PosMLP/TCN)
HigherLowerEasy                   0.521+/-0.005n3  0.518+/-0.001n3        0.529          0.529 (GRU)
HigherLowerMedium                 0.518+/-0.001n3  0.519+/-0.001n3        0.511       0.513 (IndRNN)
HigherLowerHard                   0.518+/-0.003n3  0.516+/-0.005n3        0.506       0.509 (IndRNN)
MultiarmedBanditEasy              0.281+/-0.069n3  0.327+/-0.028n3        0.619        0.631 (Elman)
MultiarmedBanditMedium            0.118+/-0.108n3  0.048+/-0.005n3        0.538          0.598 (TCN)
MultiarmedBanditHard              0.032+/-0.001n3  0.030+/-0.005n3        0.516          0.574 (TCN)
NoisyPositionOnlyCartPoleEasy     1.000+/-0.000n3  1.000+/-0.000n2        0.995          0.995 (GRU)
NoisyPositionOnlyCartPoleMedium   0.678+/-0.014n3  0.693+/-0.001n2        0.642       0.659 (IndRNN)
NoisyPositionOnlyCartPoleHard     0.367+/-0.010n3  0.373+/-0.008n2        0.390       0.404 (IndRNN)
RepeatFirstEasy                   0.986+/-0.012n3  0.860+/-0.182n3        1.000   1.000 (GRU/others)
RepeatFirstMedium                -0.393+/-0.043n3 -0.419+/-0.069n3        1.000          1.000 (GRU)
RepeatFirstHard                  -0.184+/-0.049n3 -0.306+/-0.071n3        0.940       0.969 (IndRNN)
RepeatPreviousEasy                0.695+/-0.217n3  0.890+/-0.082n3        1.000   1.000 (GRU/others)
RepeatPreviousMedium             -0.465+/-0.006n3 -0.461+/-0.007n3       -0.315          0.789 (LMU)
RepeatPreviousHard               -0.469+/-0.003n3 -0.470+/-0.001n3       -0.428          0.191 (LMU)
PositionOnlyCartPoleEasy          1.000+/-0.000n3  1.000+/-0.000n2        1.000   1.000 (GRU/others)
PositionOnlyCartPoleMedium        1.000+/-0.000n3  1.000+/-0.000n2        1.000   1.000 (GRU/others)
PositionOnlyCartPoleHard          1.000+/-0.000n3  1.000+/-0.000n2        1.000   1.000 (GRU/others)

(min steps per cell: 3.0M .. 3.0M; paper = 15M, 3 trials)
```

## Split-CIFAR-100, 3 class-order seeds (flycritic.cl.results)

```
method                       seeds  class-inc ACC  forgetting     BWT  task-inc ACC
cl_offline                       3     67.4 +/- 0.1        10.3    -9.9          90.6
cl_replay2000                    3     54.6 +/- 0.6        35.3   -35.3          87.9
cl_flymodel                      3     42.6 +/- 0.2        15.2   -15.2          77.9
cl_replay200                     3     27.2 +/- 1.3        69.5   -69.5          82.5
cl_ewc10                         3      9.1 +/- 0.1        90.3   -90.3          79.8
cl_ewc1                          3      9.1 +/- 0.1        90.3   -90.3          84.7
cl_finetune                      3      9.1 +/- 0.1        90.3   -90.3          84.6
cl_ewc100                        3      9.1 +/- 0.1        90.3   -90.3          80.9
cl_flycritic_collapsed           3      8.4 +/- 0.3        64.0   -64.0          69.6
cl_flycritic                     3      7.4 +/- 0.1        65.6   -65.6          69.7
cl_flycritic_gain                3      7.4 +/- 0.1        65.7   -65.7          69.6

Reference points (class-incremental Split-CIFAR-100, 10 tasks; APPROXIMATE, from the cited papers' tables; protocols
differ from ours in a key way: they train the ResNet-18 backbone from scratch, we use a FROZEN ImageNet ResNet-18 and
compare classifiers/memories on identical features, which makes all our numbers higher and the comparison fairer to
the Hebbian methods). Verify against the papers before quoting.
  SGD fine-tune ~8-9% ACC; online EWC ~8-9%; ER buffer 200 ~20-25%, buffer 5120 ~45-50%; DER++ buffer 5120 ~60%.
    (Buzzega et al. 2020, 'Dark Experience for General Continual Learning', NeurIPS; Boschini et al. 2022, Mammoth.)
  FlyModel-style sparse-coding + Hebbian on frozen/pretrained features: reported strong low-forgetting on Split-MNIST /
    Split-CIFAR-100 (Shen, Dasgupta & Navlakha 2021, PNAS 118(38)); exact CIFAR-100 numbers depend on their feature
    pipeline -- treat as qualitative.
Upper bound in OUR setting = 'offline' (joint linear head on frozen features), typically ~65-70% on CIFAR-100.
```
