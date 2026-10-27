# Pilot: pilot_k008_downstream_b0.0001_lr0.0001

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k008_downstream_b0.0001_lr0.0001",
  "cell": "k008_downstream_b0.0001",
  "label": "KL + 0.0001 downstream MSE",
  "k": 8,
  "lr": 0.0001,
  "seed": 900,
  "state": {
    "cell": {
      "k": 8,
      "objective": "downstream",
      "beta": 0.0001
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "complete",
    "elapsed_seconds": 59.921999999991385,
    "train_seconds": 57.110000000015134,
    "dev_seconds": 2.5160000000032596,
    "peak_vram_bytes": 1199974912,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": {
    "kl": 2.5398614406585693,
    "ce_delta": 2.5630953311920166,
    "accuracy": 0.2199999988079071,
    "teacher_accuracy": 0.9919999837875366,
    "agreement": 0.2199999988079071,
    "images": 250,
    "layer_mse": [
      [
        1.5166642665863037,
        1.2137677669525146,
        13.390203475952148
      ],
      [
        4.857597351074219,
        4.626771926879883,
        13.905935287475586
      ],
      [
        10.447400093078613,
        10.282671928405762,
        16.904603958129883
      ],
      [
        29.2935848236084,
        29.47411346435547,
        22.216108322143555
      ],
      [
        76.7779541015625,
        77.59495544433594,
        44.75117492675781
      ],
      [
        229.3022918701172,
        231.63290405273438,
        137.94407653808594
      ],
      [
        1296.148681640625,
        1305.98046875,
        910.745849609375
      ]
    ],
    "layer_nmse": [
      [
        0.8420243859291077,
        0.8531795740127563,
        0.8046417832374573
      ],
      [
        1.411725640296936,
        1.4958958625793457,
        0.8142030835151672
      ],
      [
        1.3372267484664917,
        1.364188313484192,
        0.9089111685752869
      ],
      [
        1.4773266315460205,
        1.4894379377365112,
        1.0382492542266846
      ],
      [
        1.5799658298492432,
        1.5829980373382568,
        1.397926688194275
      ],
      [
        1.5019348859786987,
        1.4974318742752075,
        1.8726022243499756
      ],
      [
        1.9278286695480347,
        1.9202756881713867,
        2.4750514030456543
      ]
    ],
    "insertion_nmse": 0.8420244455337524,
    "downstream_mse": 274.4713134765625,
    "epoch": 1,
    "images_seen": 510
  },
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 3.153047025203705,
      "kl": 3.1252062916755676,
      "insertion_nmse": 0.8621038347482681,
      "downstream_mse": 278.4066467285156,
      "lr": 9.913533761814537e-05,
      "gradient_norm": 8.631329536437988,
      "elapsed_seconds": 3.6719999999913853
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 2.9513888359069824,
      "kl": 2.923599362373352,
      "insertion_nmse": 0.7774579524993896,
      "downstream_mse": 277.89449310302734,
      "lr": 9.657457896300791e-05,
      "gradient_norm": 9.179954528808594,
      "elapsed_seconds": 7.219000000011874
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 2.619198262691498,
      "kl": 2.5919520258903503,
      "insertion_nmse": 0.8689079582691193,
      "downstream_mse": 272.4618225097656,
      "lr": 9.241613255361455e-05,
      "gradient_norm": 8.06006145477295,
      "elapsed_seconds": 10.812999999994645
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 2.8086625933647156,
      "kl": 2.780954957008362,
      "insertion_nmse": 0.7973885089159012,
      "downstream_mse": 277.07657623291016,
      "lr": 8.681980515339464e-05,
      "gradient_norm": 8.63349723815918,
      "elapsed_seconds": 14.437999999994645
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 3.344112753868103,
      "kl": 3.3164910674095154,
      "insertion_nmse": 0.8640536069869995,
      "downstream_mse": 276.21666717529297,
      "lr": 8.000066048588211e-05,
      "gradient_norm": 8.509354591369629,
      "elapsed_seconds": 17.969000000011874
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 3.09295117855072,
      "kl": 3.065528988838196,
      "insertion_nmse": 0.7790437191724777,
      "downstream_mse": 274.22174072265625,
      "lr": 7.222075445642904e-05,
      "gradient_norm": 8.48326301574707,
      "elapsed_seconds": 21.562999999994645
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 3.2884650826454163,
      "kl": 3.260762333869934,
      "insertion_nmse": 0.868651270866394,
      "downstream_mse": 277.0271301269531,
      "lr": 6.377906449072578e-05,
      "gradient_norm": 8.496750831604004,
      "elapsed_seconds": 25.094000000011874
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 2.580213576555252,
      "kl": 2.5534444451332092,
      "insertion_nmse": 0.815558448433876,
      "downstream_mse": 267.69144439697266,
      "lr": 5.500000000000001e-05,
      "gradient_norm": 8.372507095336914,
      "elapsed_seconds": 28.703999999997905
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 3.4598875045776367,
      "kl": 3.432681977748871,
      "insertion_nmse": 0.7238652110099792,
      "downstream_mse": 272.0556640625,
      "lr": 4.6220935509274235e-05,
      "gradient_norm": 8.820252418518066,
      "elapsed_seconds": 32.296999999991385
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 2.99546879529953,
      "kl": 2.9676361083984375,
      "insertion_nmse": 0.7887234836816788,
      "downstream_mse": 278.3272476196289,
      "lr": 3.777924554357096e-05,
      "gradient_norm": 9.515602111816406,
      "elapsed_seconds": 35.828999999997905
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 2.8164681792259216,
      "kl": 2.789413630962372,
      "insertion_nmse": 0.8042791038751602,
      "downstream_mse": 270.5453338623047,
      "lr": 2.9999339514117912e-05,
      "gradient_norm": 7.214019298553467,
      "elapsed_seconds": 39.40700000000652
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 2.5870742201805115,
      "kl": 2.559045970439911,
      "insertion_nmse": 0.8181268125772476,
      "downstream_mse": 280.2821350097656,
      "lr": 2.3180194846605367e-05,
      "gradient_norm": 7.823988437652588,
      "elapsed_seconds": 42.98499999998603
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 2.765313446521759,
      "kl": 2.737199127674103,
      "insertion_nmse": 0.7584545761346817,
      "downstream_mse": 281.14295196533203,
      "lr": 1.758386744638546e-05,
      "gradient_norm": 8.198381423950195,
      "elapsed_seconds": 46.51600000000326
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 2.795973479747772,
      "kl": 2.768698513507843,
      "insertion_nmse": 0.8164839148521423,
      "downstream_mse": 272.74935150146484,
      "lr": 1.3425421036992098e-05,
      "gradient_norm": 8.795575141906738,
      "elapsed_seconds": 50.15700000000652
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 2.6689953804016113,
      "kl": 2.641918271780014,
      "insertion_nmse": 0.8399732261896133,
      "downstream_mse": 270.7710189819336,
      "lr": 1.0864662381854632e-05,
      "gradient_norm": 8.33482551574707,
      "elapsed_seconds": 53.75
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 2.967508300145467,
      "kl": 2.9395100275675454,
      "insertion_nmse": 0.9088621536890665,
      "downstream_mse": 279.98284505208335,
      "lr": 1e-05,
      "gradient_norm": 8.849309921264648,
      "elapsed_seconds": 57.203999999997905
    }
  ]
}
```
