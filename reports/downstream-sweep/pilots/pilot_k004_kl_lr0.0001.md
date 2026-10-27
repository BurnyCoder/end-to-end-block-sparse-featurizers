# Pilot: pilot_k004_kl_lr0.0001

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k004_kl_lr0.0001",
  "cell": "k004_kl",
  "label": "Output KL",
  "k": 4,
  "lr": 0.0001,
  "seed": 900,
  "state": {
    "cell": {
      "k": 4,
      "objective": "kl",
      "beta": 0.0
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "complete",
    "elapsed_seconds": 79.82800000000861,
    "train_seconds": 76.95300000000861,
    "dev_seconds": 2.562000000005355,
    "peak_vram_bytes": 1215220224,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": {
    "kl": 3.3337953090667725,
    "ce_delta": 3.3342607021331787,
    "accuracy": 0.14000000059604645,
    "teacher_accuracy": 0.9919999837875366,
    "agreement": 0.13600000739097595,
    "images": 250,
    "layer_mse": [
      [
        1.6237698793411255,
        1.2993433475494385,
        14.34127140045166
      ],
      [
        5.7067084312438965,
        5.470071315765381,
        14.982939720153809
      ],
      [
        11.712143898010254,
        11.522322654724121,
        19.153209686279297
      ],
      [
        29.975522994995117,
        30.11995506286621,
        24.313657760620117
      ],
      [
        79.30559539794922,
        80.06471252441406,
        49.54905319213867
      ],
      [
        237.23973083496094,
        239.39431762695312,
        152.78074645996094
      ],
      [
        1529.2086181640625,
        1538.2071533203125,
        1176.4788818359375
      ]
    ],
    "layer_nmse": [
      [
        0.9014875888824463,
        0.9133318662643433,
        0.8617929220199585
      ],
      [
        1.6584964990615845,
        1.768546462059021,
        0.87726229429245
      ],
      [
        1.4991093873977661,
        1.5286496877670288,
        1.0298120975494385
      ],
      [
        1.511717677116394,
        1.5220738649368286,
        1.1362754106521606
      ],
      [
        1.6319807767868042,
        1.633382797241211,
        1.5478010177612305
      ],
      [
        1.5539249181747437,
        1.5476065874099731,
        2.0740108489990234
      ],
      [
        2.2744712829589844,
        2.261734962463379,
        3.1972124576568604
      ]
    ],
    "insertion_nmse": 0.9014871716499329,
    "downstream_mse": 315.52484130859375,
    "epoch": 1,
    "images_seen": 510
  },
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 3.3756935000419617,
      "kl": 3.3756935000419617,
      "insertion_nmse": 0.9228818565607071,
      "downstream_mse": 320.08287048339844,
      "lr": 9.913533761814537e-05,
      "gradient_norm": 5.058109283447266,
      "elapsed_seconds": 5.562000000005355
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 3.527978301048279,
      "kl": 3.527978301048279,
      "insertion_nmse": 0.8320979624986649,
      "downstream_mse": 318.3322067260742,
      "lr": 9.657457896300791e-05,
      "gradient_norm": 5.29170560836792,
      "elapsed_seconds": 10.35899999999674
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 3.585894227027893,
      "kl": 3.585894227027893,
      "insertion_nmse": 0.930299237370491,
      "downstream_mse": 317.5802688598633,
      "lr": 9.241613255361455e-05,
      "gradient_norm": 5.106945514678955,
      "elapsed_seconds": 15.344000000011874
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 3.324130117893219,
      "kl": 3.324130117893219,
      "insertion_nmse": 0.854008138179779,
      "downstream_mse": 318.5211715698242,
      "lr": 8.681980515339464e-05,
      "gradient_norm": 5.161124229431152,
      "elapsed_seconds": 20.125
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 4.279997825622559,
      "kl": 4.279997825622559,
      "insertion_nmse": 0.9233495146036148,
      "downstream_mse": 321.74732208251953,
      "lr": 8.000066048588211e-05,
      "gradient_norm": 5.568781852722168,
      "elapsed_seconds": 24.875
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 3.7155675888061523,
      "kl": 3.7155675888061523,
      "insertion_nmse": 0.8329456448554993,
      "downstream_mse": 314.07381439208984,
      "lr": 7.222075445642904e-05,
      "gradient_norm": 5.077180862426758,
      "elapsed_seconds": 29.76600000000326
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 3.186128556728363,
      "kl": 3.186128556728363,
      "insertion_nmse": 0.9305736720561981,
      "downstream_mse": 318.21112060546875,
      "lr": 6.377906449072578e-05,
      "gradient_norm": 5.089451789855957,
      "elapsed_seconds": 34.953000000008615
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 3.473091244697571,
      "kl": 3.473091244697571,
      "insertion_nmse": 0.8729769289493561,
      "downstream_mse": 309.1990737915039,
      "lr": 5.500000000000001e-05,
      "gradient_norm": 5.673215866088867,
      "elapsed_seconds": 39.875
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 4.12212872505188,
      "kl": 4.12212872505188,
      "insertion_nmse": 0.774670422077179,
      "downstream_mse": 314.78829193115234,
      "lr": 4.6220935509274235e-05,
      "gradient_norm": 5.626572608947754,
      "elapsed_seconds": 45.078000000008615
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 3.3092848658561707,
      "kl": 3.3092848658561707,
      "insertion_nmse": 0.8444958329200745,
      "downstream_mse": 319.3863983154297,
      "lr": 3.777924554357096e-05,
      "gradient_norm": 5.5902018547058105,
      "elapsed_seconds": 50.094000000011874
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 3.3504027128219604,
      "kl": 3.3504027128219604,
      "insertion_nmse": 0.8610584586858749,
      "downstream_mse": 313.4277038574219,
      "lr": 2.9999339514117912e-05,
      "gradient_norm": 4.790074825286865,
      "elapsed_seconds": 55.219000000011874
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 3.2487305998802185,
      "kl": 3.2487305998802185,
      "insertion_nmse": 0.8744997084140778,
      "downstream_mse": 318.50543212890625,
      "lr": 2.3180194846605367e-05,
      "gradient_norm": 5.688134670257568,
      "elapsed_seconds": 60.469000000011874
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 3.2884182929992676,
      "kl": 3.2884182929992676,
      "insertion_nmse": 0.8111472427845001,
      "downstream_mse": 323.8027648925781,
      "lr": 1.758386744638546e-05,
      "gradient_norm": 5.222769260406494,
      "elapsed_seconds": 65.0
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 3.39383327960968,
      "kl": 3.39383327960968,
      "insertion_nmse": 0.8738299310207367,
      "downstream_mse": 313.7021713256836,
      "lr": 1.3425421036992098e-05,
      "gradient_norm": 5.772282123565674,
      "elapsed_seconds": 69.28100000001723
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 3.246495842933655,
      "kl": 3.246495842933655,
      "insertion_nmse": 0.8992981165647507,
      "downstream_mse": 311.3175354003906,
      "lr": 1.0864662381854632e-05,
      "gradient_norm": 5.489067077636719,
      "elapsed_seconds": 73.26600000000326
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 3.691527509689331,
      "kl": 3.691527509689331,
      "insertion_nmse": 0.9720762213071187,
      "downstream_mse": 321.4348571777344,
      "lr": 1e-05,
      "gradient_norm": 6.06298303604126,
      "elapsed_seconds": 77.06200000000536
    }
  ]
}
```
