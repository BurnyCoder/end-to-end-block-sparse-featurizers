# Pilot: pilot_k004_downstream_b0.0001_lr0.0001

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k004_downstream_b0.0001_lr0.0001",
  "cell": "k004_downstream_b0.0001",
  "label": "KL + 0.0001 downstream MSE",
  "k": 4,
  "lr": 0.0001,
  "seed": 900,
  "state": {
    "cell": {
      "k": 4,
      "objective": "downstream",
      "beta": 0.0001
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "complete",
    "elapsed_seconds": 66.625,
    "train_seconds": 63.74900000001071,
    "dev_seconds": 2.625,
    "peak_vram_bytes": 1199974912,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": {
    "kl": 3.3340749740600586,
    "ce_delta": 3.335883140563965,
    "accuracy": 0.13600000739097595,
    "teacher_accuracy": 0.9919999837875366,
    "agreement": 0.13199999928474426,
    "images": 250,
    "layer_mse": [
      [
        1.6237914562225342,
        1.299344778060913,
        14.342080116271973
      ],
      [
        5.707393169403076,
        5.470747947692871,
        14.983780860900879
      ],
      [
        11.712672233581543,
        11.5228271484375,
        19.154592514038086
      ],
      [
        29.974714279174805,
        30.119104385375977,
        24.314653396606445
      ],
      [
        79.30162811279297,
        80.06085205078125,
        49.53998565673828
      ],
      [
        237.2158660888672,
        239.37013244628906,
        152.76463317871094
      ],
      [
        1529.095703125,
        1538.0955810546875,
        1176.3001708984375
      ]
    ],
    "layer_nmse": [
      [
        0.9014986157417297,
        0.9133321046829224,
        0.8618414998054504
      ],
      [
        1.6586953401565552,
        1.768764853477478,
        0.8773114681243896
      ],
      [
        1.499177098274231,
        1.5287176370620728,
        1.0298867225646973
      ],
      [
        1.5116769075393677,
        1.5220310688018799,
        1.1363229751586914
      ],
      [
        1.6318987607955933,
        1.6333050727844238,
        1.5475177764892578
      ],
      [
        1.553768515586853,
        1.5474516153335571,
        2.0737924575805664
      ],
      [
        2.2743027210235596,
        2.261570692062378,
        3.196725368499756
      ]
    ],
    "insertion_nmse": 0.9014986753463745,
    "downstream_mse": 315.5013732910156,
    "epoch": 1,
    "images_seen": 510
  },
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 3.407701790332794,
      "kl": 3.3756935000419617,
      "insertion_nmse": 0.9228818565607071,
      "downstream_mse": 320.08287048339844,
      "lr": 9.913533761814537e-05,
      "gradient_norm": 5.058083534240723,
      "elapsed_seconds": 4.187000000005355
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 3.5585652589797974,
      "kl": 3.5267322659492493,
      "insertion_nmse": 0.832098513841629,
      "downstream_mse": 318.3299102783203,
      "lr": 9.657457896300791e-05,
      "gradient_norm": 5.2909088134765625,
      "elapsed_seconds": 8.264999999984866
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 3.617290735244751,
      "kl": 3.5855363607406616,
      "insertion_nmse": 0.9302992224693298,
      "downstream_mse": 317.5439758300781,
      "lr": 9.241613255361455e-05,
      "gradient_norm": 5.085901260375977,
      "elapsed_seconds": 12.327999999979511
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 3.3507569432258606,
      "kl": 3.318917453289032,
      "insertion_nmse": 0.8540049493312836,
      "downstream_mse": 318.3944854736328,
      "lr": 8.681980515339464e-05,
      "gradient_norm": 5.150817394256592,
      "elapsed_seconds": 16.23399999999674
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 4.31530636548996,
      "kl": 4.283128440380096,
      "insertion_nmse": 0.923349991440773,
      "downstream_mse": 321.77882385253906,
      "lr": 8.000066048588211e-05,
      "gradient_norm": 5.524509429931641,
      "elapsed_seconds": 20.25
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 3.756759822368622,
      "kl": 3.7253660559654236,
      "insertion_nmse": 0.8329458981752396,
      "downstream_mse": 313.9374313354492,
      "lr": 7.222075445642904e-05,
      "gradient_norm": 5.065542697906494,
      "elapsed_seconds": 24.280999999988126
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 3.216367185115814,
      "kl": 3.184549331665039,
      "insertion_nmse": 0.9305773377418518,
      "downstream_mse": 318.17838287353516,
      "lr": 6.377906449072578e-05,
      "gradient_norm": 5.099277496337891,
      "elapsed_seconds": 28.312000000005355
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 3.4882792830467224,
      "kl": 3.457354187965393,
      "insertion_nmse": 0.8729789108037949,
      "downstream_mse": 309.25098419189453,
      "lr": 5.500000000000001e-05,
      "gradient_norm": 5.636220455169678,
      "elapsed_seconds": 32.264999999984866
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 4.162832319736481,
      "kl": 4.131352186203003,
      "insertion_nmse": 0.7746753543615341,
      "downstream_mse": 314.8013610839844,
      "lr": 4.6220935509274235e-05,
      "gradient_norm": 5.612499713897705,
      "elapsed_seconds": 36.389999999984866
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 3.337853252887726,
      "kl": 3.3058992624282837,
      "insertion_nmse": 0.844500869512558,
      "downstream_mse": 319.53985595703125,
      "lr": 3.777924554357096e-05,
      "gradient_norm": 5.6033101081848145,
      "elapsed_seconds": 40.405999999988126
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 3.3889631628990173,
      "kl": 3.3576215505599976,
      "insertion_nmse": 0.8610668629407883,
      "downstream_mse": 313.4163818359375,
      "lr": 2.9999339514117912e-05,
      "gradient_norm": 4.8213324546813965,
      "elapsed_seconds": 44.32799999997951
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 3.286580801010132,
      "kl": 3.254742681980133,
      "insertion_nmse": 0.8745023459196091,
      "downstream_mse": 318.3807373046875,
      "lr": 2.3180194846605367e-05,
      "gradient_norm": 5.64459753036499,
      "elapsed_seconds": 48.155999999988126
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 3.3160961866378784,
      "kl": 3.283726155757904,
      "insertion_nmse": 0.8111549317836761,
      "downstream_mse": 323.70054626464844,
      "lr": 1.758386744638546e-05,
      "gradient_norm": 5.231685638427734,
      "elapsed_seconds": 52.171999999991385
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 3.428607225418091,
      "kl": 3.397258996963501,
      "insertion_nmse": 0.8738310039043427,
      "downstream_mse": 313.48206329345703,
      "lr": 1.3425421036992098e-05,
      "gradient_norm": 5.8345770835876465,
      "elapsed_seconds": 55.98399999999674
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 3.264513850212097,
      "kl": 3.2333873510360718,
      "insertion_nmse": 0.8993033915758133,
      "downstream_mse": 311.2644271850586,
      "lr": 1.0864662381854632e-05,
      "gradient_norm": 5.5290608406066895,
      "elapsed_seconds": 59.98399999999674
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 3.733495251337687,
      "kl": 3.7013655503590903,
      "insertion_nmse": 0.972088102499644,
      "downstream_mse": 321.2963033040365,
      "lr": 1e-05,
      "gradient_norm": 6.060658931732178,
      "elapsed_seconds": 63.84299999999348
    }
  ]
}
```
