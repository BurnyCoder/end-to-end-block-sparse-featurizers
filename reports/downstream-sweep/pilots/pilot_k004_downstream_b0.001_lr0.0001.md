# Pilot: pilot_k004_downstream_b0.001_lr0.0001

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k004_downstream_b0.001_lr0.0001",
  "cell": "k004_downstream_b0.001",
  "label": "KL + 0.001 downstream MSE",
  "k": 4,
  "lr": 0.0001,
  "seed": 900,
  "state": {
    "cell": {
      "k": 4,
      "objective": "downstream",
      "beta": 0.001
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "complete",
    "elapsed_seconds": 56.48499999998603,
    "train_seconds": 53.812000000005355,
    "dev_seconds": 2.4059999999881256,
    "peak_vram_bytes": 1199974912,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": {
    "kl": 3.330404043197632,
    "ce_delta": 3.3322465419769287,
    "accuracy": 0.13600000739097595,
    "teacher_accuracy": 0.9919999837875366,
    "agreement": 0.13199999928474426,
    "images": 250,
    "layer_mse": [
      [
        1.6237705945968628,
        1.2993391752243042,
        14.341485977172852
      ],
      [
        5.705648899078369,
        5.468983173370361,
        14.982935905456543
      ],
      [
        11.712231636047363,
        11.522429466247559,
        19.15261459350586
      ],
      [
        29.97391128540039,
        30.118295669555664,
        24.31432342529297
      ],
      [
        79.27200317382812,
        80.03067016601562,
        49.53398132324219
      ],
      [
        237.04364013671875,
        239.19390869140625,
        152.74850463867188
      ],
      [
        1527.0006103515625,
        1536.0028076171875,
        1174.0943603515625
      ]
    ],
    "layer_nmse": [
      [
        0.9014881253242493,
        0.9133287072181702,
        0.8618058562278748
      ],
      [
        1.6581889390945435,
        1.768194556236267,
        0.8772620558738708
      ],
      [
        1.499122142791748,
        1.528663992881775,
        1.0297799110412598
      ],
      [
        1.5116368532180786,
        1.5219895839691162,
        1.1363065242767334
      ],
      [
        1.6312898397445679,
        1.6326874494552612,
        1.547330617904663
      ],
      [
        1.5526399612426758,
        1.5463107824325562,
        2.073573112487793
      ],
      [
        2.2711853981018066,
        2.2584939002990723,
        3.1907317638397217
      ]
    ],
    "insertion_nmse": 0.9014875888824463,
    "downstream_mse": 315.117919921875,
    "epoch": 1,
    "images_seen": 510
  },
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 3.6957764625549316,
      "kl": 3.3756935000419617,
      "insertion_nmse": 0.9228818565607071,
      "downstream_mse": 320.08287048339844,
      "lr": 9.913533761814537e-05,
      "gradient_norm": 5.058178901672363,
      "elapsed_seconds": 3.60999999998603
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 3.845050811767578,
      "kl": 3.52674663066864,
      "insertion_nmse": 0.8320997953414917,
      "downstream_mse": 318.30413818359375,
      "lr": 9.657457896300791e-05,
      "gradient_norm": 5.276377201080322,
      "elapsed_seconds": 6.952999999979511
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 3.9166818857192993,
      "kl": 3.5992212295532227,
      "insertion_nmse": 0.9302928447723389,
      "downstream_mse": 317.4605941772461,
      "lr": 9.241613255361455e-05,
      "gradient_norm": 5.100414752960205,
      "elapsed_seconds": 10.796999999991385
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 3.6347145438194275,
      "kl": 3.316294491291046,
      "insertion_nmse": 0.8540028780698776,
      "downstream_mse": 318.42003631591797,
      "lr": 8.681980515339464e-05,
      "gradient_norm": 5.136097431182861,
      "elapsed_seconds": 14.375
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 4.608196258544922,
      "kl": 4.28651350736618,
      "insertion_nmse": 0.9233463704586029,
      "downstream_mse": 321.6827621459961,
      "lr": 8.000066048588211e-05,
      "gradient_norm": 5.5664567947387695,
      "elapsed_seconds": 17.640999999974156
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 4.029138267040253,
      "kl": 3.715400815010071,
      "insertion_nmse": 0.832942470908165,
      "downstream_mse": 313.7374572753906,
      "lr": 7.222075445642904e-05,
      "gradient_norm": 5.123580455780029,
      "elapsed_seconds": 20.875
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 3.5083683729171753,
      "kl": 3.190238118171692,
      "insertion_nmse": 0.9305692166090012,
      "downstream_mse": 318.1301574707031,
      "lr": 6.377906449072578e-05,
      "gradient_norm": 5.103243827819824,
      "elapsed_seconds": 24.062999999994645
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 3.7722955346107483,
      "kl": 3.4633949995040894,
      "insertion_nmse": 0.8729724735021591,
      "downstream_mse": 308.9005355834961,
      "lr": 5.500000000000001e-05,
      "gradient_norm": 5.625670909881592,
      "elapsed_seconds": 27.265999999974156
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 4.439270734786987,
      "kl": 4.124905228614807,
      "insertion_nmse": 0.7746509462594986,
      "downstream_mse": 314.3653869628906,
      "lr": 4.6220935509274235e-05,
      "gradient_norm": 5.619251251220703,
      "elapsed_seconds": 30.437999999994645
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 3.6301543712615967,
      "kl": 3.311281681060791,
      "insertion_nmse": 0.844494417309761,
      "downstream_mse": 318.87259674072266,
      "lr": 3.777924554357096e-05,
      "gradient_norm": 5.614629745483398,
      "elapsed_seconds": 34.030999999988126
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 3.6687700152397156,
      "kl": 3.355631411075592,
      "insertion_nmse": 0.8610631972551346,
      "downstream_mse": 313.13865661621094,
      "lr": 2.9999339514117912e-05,
      "gradient_norm": 4.762275695800781,
      "elapsed_seconds": 37.515999999974156
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 3.565960943698883,
      "kl": 3.247975707054138,
      "insertion_nmse": 0.8744871318340302,
      "downstream_mse": 317.9851531982422,
      "lr": 2.3180194846605367e-05,
      "gradient_norm": 5.620230674743652,
      "elapsed_seconds": 40.937999999994645
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 3.6200706362724304,
      "kl": 3.296792447566986,
      "insertion_nmse": 0.8111411184072495,
      "downstream_mse": 323.27821350097656,
      "lr": 1.758386744638546e-05,
      "gradient_norm": 5.2745561599731445,
      "elapsed_seconds": 44.10999999998603
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 3.705796182155609,
      "kl": 3.3925201892852783,
      "insertion_nmse": 0.8738309442996979,
      "downstream_mse": 313.2760009765625,
      "lr": 1.3425421036992098e-05,
      "gradient_norm": 5.785531044006348,
      "elapsed_seconds": 47.25
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 3.5466803908348083,
      "kl": 3.2357919812202454,
      "insertion_nmse": 0.8993007391691208,
      "downstream_mse": 310.8884506225586,
      "lr": 1.0864662381854632e-05,
      "gradient_norm": 5.4903388023376465,
      "elapsed_seconds": 50.562999999994645
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 4.026523240407308,
      "kl": 3.705850346883138,
      "insertion_nmse": 0.9720717390378316,
      "downstream_mse": 320.672890218099,
      "lr": 1e-05,
      "gradient_norm": 6.052929878234863,
      "elapsed_seconds": 53.905999999988126
    }
  ]
}
```
