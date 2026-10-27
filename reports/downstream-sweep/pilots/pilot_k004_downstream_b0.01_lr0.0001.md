# Pilot: pilot_k004_downstream_b0.01_lr0.0001

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k004_downstream_b0.01_lr0.0001",
  "cell": "k004_downstream_b0.01",
  "label": "KL + 0.01 downstream MSE",
  "k": 4,
  "lr": 0.0001,
  "seed": 900,
  "state": {
    "cell": {
      "k": 4,
      "objective": "downstream",
      "beta": 0.01
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "complete",
    "elapsed_seconds": 61.125,
    "train_seconds": 58.01500000007218,
    "dev_seconds": 2.4839999999967404,
    "peak_vram_bytes": 1199974912,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": {
    "kl": 3.2924516201019287,
    "ce_delta": 3.294750928878784,
    "accuracy": 0.15199999511241913,
    "teacher_accuracy": 0.9919999837875366,
    "agreement": 0.14800000190734863,
    "images": 250,
    "layer_mse": [
      [
        1.6235849857330322,
        1.2992464303970337,
        14.337655067443848
      ],
      [
        5.6972198486328125,
        5.460453033447266,
        14.978453636169434
      ],
      [
        11.704160690307617,
        11.51443862915039,
        19.141338348388672
      ],
      [
        29.943653106689453,
        30.087688446044922,
        24.2976131439209
      ],
      [
        79.04832458496094,
        79.8032455444336,
        49.45751190185547
      ],
      [
        235.89479064941406,
        238.02398681640625,
        152.43075561523438
      ],
      [
        1510.9898681640625,
        1520.006103515625,
        1157.5413818359375
      ]
    ],
    "layer_nmse": [
      [
        0.9013846516609192,
        0.913263738155365,
        0.8615753650665283
      ],
      [
        1.6557378768920898,
        1.7654355764389038,
        0.8769994974136353
      ],
      [
        1.4980876445770264,
        1.527604103088379,
        1.0291740894317627
      ],
      [
        1.510110855102539,
        1.520443320274353,
        1.1355259418487549
      ],
      [
        1.6266865730285645,
        1.6280484199523926,
        1.5449403524398804
      ],
      [
        1.5451158285140991,
        1.5387486219406128,
        2.069260597229004
      ],
      [
        2.247372627258301,
        2.234971761703491,
        3.1457457542419434
      ]
    ],
    "insertion_nmse": 0.9013846516609192,
    "downstream_mse": 312.21307373046875,
    "epoch": 1,
    "images_seen": 510
  },
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 6.57652223110199,
      "kl": 3.3756935000419617,
      "insertion_nmse": 0.9228818565607071,
      "downstream_mse": 320.08287048339844,
      "lr": 9.913533761814537e-05,
      "gradient_norm": 5.091587066650391,
      "elapsed_seconds": 3.687000000005355
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 6.734570384025574,
      "kl": 3.553529739379883,
      "insertion_nmse": 0.8320876955986023,
      "downstream_mse": 318.1040573120117,
      "lr": 9.657457896300791e-05,
      "gradient_norm": 5.3332672119140625,
      "elapsed_seconds": 7.342999999993481
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 6.747730493545532,
      "kl": 3.5783985257148743,
      "insertion_nmse": 0.9302683770656586,
      "downstream_mse": 316.9332046508789,
      "lr": 9.241613255361455e-05,
      "gradient_norm": 5.219002723693848,
      "elapsed_seconds": 11.030999999988126
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 6.4838080406188965,
      "kl": 3.3072535395622253,
      "insertion_nmse": 0.8539707809686661,
      "downstream_mse": 317.6554489135742,
      "lr": 8.681980515339464e-05,
      "gradient_norm": 5.295687198638916,
      "elapsed_seconds": 14.546999999991385
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 7.465574622154236,
      "kl": 4.26039445400238,
      "insertion_nmse": 0.9232877641916275,
      "downstream_mse": 320.5180206298828,
      "lr": 8.000066048588211e-05,
      "gradient_norm": 5.650577545166016,
      "elapsed_seconds": 18.264999999984866
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 6.816267490386963,
      "kl": 3.6883984804153442,
      "insertion_nmse": 0.8328966200351715,
      "downstream_mse": 312.78690338134766,
      "lr": 7.222075445642904e-05,
      "gradient_norm": 5.3191423416137695,
      "elapsed_seconds": 21.889999999984866
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 6.357312202453613,
      "kl": 3.1934270560741425,
      "insertion_nmse": 0.9305002093315125,
      "downstream_mse": 316.3885269165039,
      "lr": 6.377906449072578e-05,
      "gradient_norm": 5.089509963989258,
      "elapsed_seconds": 25.562000000005355
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 6.526722550392151,
      "kl": 3.4537429213523865,
      "insertion_nmse": 0.8728990703821182,
      "downstream_mse": 307.2979736328125,
      "lr": 5.500000000000001e-05,
      "gradient_norm": 5.813525199890137,
      "elapsed_seconds": 29.139999999984866
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 7.2264604568481445,
      "kl": 4.10055685043335,
      "insertion_nmse": 0.7745861262083054,
      "downstream_mse": 312.5903778076172,
      "lr": 4.6220935509274235e-05,
      "gradient_norm": 5.7497944831848145,
      "elapsed_seconds": 32.60899999999674
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 6.465399384498596,
      "kl": 3.2984121441841125,
      "insertion_nmse": 0.8444066196680069,
      "downstream_mse": 316.69873809814453,
      "lr": 3.777924554357096e-05,
      "gradient_norm": 5.831307411193848,
      "elapsed_seconds": 36.187000000005355
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 6.450921654701233,
      "kl": 3.340693175792694,
      "insertion_nmse": 0.8609766662120819,
      "downstream_mse": 311.02283477783203,
      "lr": 2.9999339514117912e-05,
      "gradient_norm": 5.032930374145508,
      "elapsed_seconds": 39.889999999984866
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 6.389954090118408,
      "kl": 3.2347952127456665,
      "insertion_nmse": 0.8743894100189209,
      "downstream_mse": 315.5158996582031,
      "lr": 2.3180194846605367e-05,
      "gradient_norm": 5.925983905792236,
      "elapsed_seconds": 43.48399999999674
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 6.447947978973389,
      "kl": 3.2416491508483887,
      "insertion_nmse": 0.8110550194978714,
      "downstream_mse": 320.62989807128906,
      "lr": 1.758386744638546e-05,
      "gradient_norm": 5.505693435668945,
      "elapsed_seconds": 47.10899999999674
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 6.471679210662842,
      "kl": 3.3655093908309937,
      "insertion_nmse": 0.873740017414093,
      "downstream_mse": 310.61700439453125,
      "lr": 1.3425421036992098e-05,
      "gradient_norm": 5.983936309814453,
      "elapsed_seconds": 50.75
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 6.294367551803589,
      "kl": 3.211847960948944,
      "insertion_nmse": 0.8992046564817429,
      "downstream_mse": 308.2519760131836,
      "lr": 1.0864662381854632e-05,
      "gradient_norm": 5.822006702423096,
      "elapsed_seconds": 54.46799999999348
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 6.877146498362223,
      "kl": 3.699445295333862,
      "insertion_nmse": 0.9719679196675618,
      "downstream_mse": 317.7701395670573,
      "lr": 1e-05,
      "gradient_norm": 6.1870574951171875,
      "elapsed_seconds": 58.125
    }
  ]
}
```
